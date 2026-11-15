// Camera -> guide-box crop -> OCR, all in this tab. Nothing is uploaded: the page only
// downloads its own code and model, then reads frames locally.
import * as ort from "./vendor/ort.wasm.min.mjs";
import { createRecognizer, isAcceptable } from "./ocr.js";

ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
ort.env.wasm.numThreads = 1; // GitHub Pages cannot send COOP/COEP headers, so no wasm threads

const $ = (id) => document.getElementById(id);
const video = $("video");
const crop = $("crop");
const cropContext = crop.getContext("2d", { willReadFrequently: true });

// Smallest valid module, and one using a v128 instruction (from wasm-feature-detect).
const WASM_EMPTY = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]);
const WASM_SIMD = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
]);

const state = { recognizer: null, busy: false, frozen: false, last: 0, interval: 500, candidate: null, stable: "" };

function showError(error) {
  $("error").textContent = error instanceof Error ? error.message : String(error);
}

async function probe() {
  const wasm = typeof WebAssembly === "object" && (await WebAssembly.instantiate(WASM_EMPTY).then(() => true, () => false));
  return [
    ["Secure context (HTTPS)", window.isSecureContext],
    ["Camera API (getUserMedia)", Boolean(navigator.mediaDevices?.getUserMedia)],
    ["WebAssembly", wasm],
    ["WebAssembly SIMD", wasm && WebAssembly.validate(WASM_SIMD)],
    ["WebGPU (not used)", "gpu" in navigator],
    ["Speech synthesis (not used yet)", "speechSynthesis" in window],
  ];
}

function report(checks, selfTest) {
  $("checklist").replaceChildren(
    ...checks.map(([name, ok]) => Object.assign(document.createElement("li"), { textContent: `${ok ? "✓" : "✗"} ${name}` })),
    Object.assign(document.createElement("li"), { textContent: `UA: ${navigator.userAgent}` }),
  );
  $("summary").textContent = selfTest;
}

async function fetchOk(path, as) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response[as]();
}

/** Reads a fixture line the Python runtime has already read, and compares. */
async function selfTest(recognizer) {
  const expected = await fetchOk("fixtures/expected.json", "json");
  const line = expected.lines[0];
  const bitmap = await createImageBitmap(await fetchOk(`fixtures/${line.file}`, "blob"));
  const canvas = Object.assign(document.createElement("canvas"), { width: bitmap.width, height: bitmap.height });
  const context = canvas.getContext("2d");
  context.drawImage(bitmap, 0, 0);
  const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
  let reading;
  for (let run = 0; run < 3; run++) reading = await recognizer.read(data, bitmap.width, bitmap.height); // last = warm
  const ms = reading.prepMs + reading.inferMs + reading.decodeMs;
  const verdict = reading.text === line.text ? "matches reference" : `MISMATCH: “${reading.text}”`;
  return `Self-test ${verdict} · ${ms.toFixed(1)} ms per line (warm)`;
}

/** The guide box in video pixels; the video is drawn with object-fit: cover. */
function guideInVideo() {
  const view = video.getBoundingClientRect();
  const guide = $("guide").getBoundingClientRect();
  const scale = Math.max(view.width / video.videoWidth, view.height / video.videoHeight);
  const offsetX = (view.width - video.videoWidth * scale) / 2;
  const offsetY = (view.height - video.videoHeight * scale) / 2;
  const x = Math.max(0, Math.round((guide.left - view.left - offsetX) / scale));
  const y = Math.max(0, Math.round((guide.top - view.top - offsetY) / scale));
  const w = Math.min(video.videoWidth - x, Math.round(guide.width / scale));
  const h = Math.min(video.videoHeight - y, Math.round(guide.height / scale));
  return { x, y, w, h };
}

/** Two acceptable readings in a row that agree; anything else resets the run. */
function confirm(reading) {
  if (!isAcceptable(reading.text, reading.confidence)) {
    state.candidate = null;
    return null;
  }
  const confirmed = reading.text === state.candidate ? reading.text : null;
  state.candidate = reading.text;
  return confirmed;
}

async function readFrame() {
  const start = performance.now();
  const { x, y, w, h } = guideInVideo();
  if (w < 1 || h < 1) return;
  crop.width = w;
  crop.height = h;
  cropContext.drawImage(video, x, y, w, h, 0, 0, w, h);
  const { data } = cropContext.getImageData(0, 0, w, h);
  const cropMs = performance.now() - start;

  const reading = await state.recognizer.read(data, w, h);
  const confirmed = confirm(reading);
  if (confirmed && confirmed !== state.stable) {
    state.stable = confirmed;
    $("stable").textContent = confirmed;
    $("copy").disabled = false;
  }
  const filtered = isAcceptable(reading.text, reading.confidence) ? "" : " (filtered)";
  $("last").textContent = `last ${reading.confidence.toFixed(2)}${filtered}: ${reading.text}`;
  $("timing").textContent =
    `${w}×${h} px · crop ${cropMs.toFixed(1)} · prep ${reading.prepMs.toFixed(1)} · ` +
    `infer ${reading.inferMs.toFixed(1)} · decode ${reading.decodeMs.toFixed(1)} · ` +
    `total ${(performance.now() - start).toFixed(1)} ms`;
  $("error").textContent = "";
}

/** One job at a time, latest frame only: frames that arrive while busy are skipped. */
function loop(now) {
  requestAnimationFrame(loop);
  if (state.busy || state.frozen || now - state.last < state.interval || video.readyState < 2) return;
  state.busy = true;
  state.last = now;
  readFrame()
    .catch(showError)
    .finally(() => {
      state.busy = false;
    });
}

async function startCamera() {
  video.srcObject = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: "environment" }, width: { ideal: 3840 }, height: { ideal: 2160 } },
  });
  await video.play();
  $("start").hidden = true;
  $("freeze").disabled = false;
  $("checks").open = false;
  requestAnimationFrame(loop);
}

$("start").onclick = () => startCamera().catch(showError);
$("freeze").onclick = () => {
  state.frozen = !state.frozen;
  $("freeze").textContent = state.frozen ? "Resume" : "Freeze";
  if (state.frozen) video.pause();
  else video.play().catch(showError);
};
$("copy").onclick = () => navigator.clipboard.writeText(state.stable).catch(showError);
$("rate").onchange = (event) => {
  state.interval = Number(event.target.value);
};

async function main() {
  const checks = await probe();
  report(checks, "Loading model…");
  if (!checks.find(([name]) => name === "WebAssembly")[1]) {
    report(checks, "WebAssembly is off, so OCR cannot run here.");
    throw new Error(
      "WebAssembly is blocked. If Lockdown Mode is on: page menu → Website Settings → turn off Lockdown Mode for this site, then reload.",
    );
  }
  const [model, dictionary] = await Promise.all([fetchOk("model/rec.onnx", "arrayBuffer"), fetchOk("model/rec_dict.txt", "text")]);
  state.recognizer = await createRecognizer(ort, model, dictionary);
  report(checks, await selfTest(state.recognizer));
  $("start").disabled = !checks.find(([name]) => name.startsWith("Camera"))[1];
}

main().catch(showError);
