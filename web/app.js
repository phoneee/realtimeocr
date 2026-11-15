// Camera -> whole-frame OCR, all in this tab. Nothing is uploaded: the page only downloads
// its own code and models, then reads frames locally.
import * as ort from "./vendor/ort.wasm.min.mjs";
import { createPageReader } from "./page.js";

ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
ort.env.wasm.numThreads = 1; // GitHub Pages cannot send COOP/COEP headers, so no wasm threads

// Frames bigger than this are scaled down before reading: bounds the per-frame copy.
const MAX_FRAME_SIDE = 2560;

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("boxes");
const frame = document.createElement("canvas");
const frameContext = frame.getContext("2d", { willReadFrequently: true });

// Smallest valid module, and one using a v128 instruction (from wasm-feature-detect).
const WASM_EMPTY = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]);
const WASM_SIMD = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
]);

const state = { reader: null, busy: false, frozen: false, limit: 960, text: "", lines: [], frameSize: [0, 0] };

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

function report(checks, summary) {
  $("checklist").replaceChildren(
    ...checks.map(([name, ok]) => Object.assign(document.createElement("li"), { textContent: `${ok ? "✓" : "✗"} ${name}` })),
    Object.assign(document.createElement("li"), { textContent: `UA: ${navigator.userAgent}` }),
  );
  $("summary").textContent = summary;
}

async function fetchOk(path, as) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response[as]();
}

/** Reads a fixture page the Python runtime has already read, and compares. */
async function selfTest(reader) {
  const expected = await fetchOk("fixtures/expected.json", "json");
  const page = expected.pages[0];
  const bitmap = await createImageBitmap(await fetchOk(`fixtures/${page.file}`, "blob"));
  const canvas = Object.assign(document.createElement("canvas"), { width: bitmap.width, height: bitmap.height });
  const context = canvas.getContext("2d");
  context.drawImage(bitmap, 0, 0);
  const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
  await reader.read(data, bitmap.width, bitmap.height, 1280); // warm-up
  const reading = await reader.read(data, bitmap.width, bitmap.height, 1280);
  const edits = editDistance([...reading.text], [...page.text]);
  return `Self-test: ${reading.lines.length}/${page.lines} lines, ${edits} of ${[...page.text].length} characters ` +
    `differ from the reference · detect ${reading.detMs.toFixed(0)} · read ${reading.recMs.toFixed(0)} ms`;
}

function editDistance(a, b) {
  let row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const next = [i];
    for (let j = 1; j <= b.length; j++) {
      next[j] = Math.min(row[j] + 1, next[j - 1] + 1, row[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    row = next;
  }
  return row[b.length];
}

/** Frame pixels -> overlay pixels, matching the video's object-fit: cover. */
function drawBoxes() {
  const { width, height } = overlay.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  overlay.width = Math.round(width * ratio);
  overlay.height = Math.round(height * ratio);
  const context = overlay.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  const [fw, fh] = state.frameSize;
  if (!fw || !fh) return;
  const scale = Math.max(width / fw, height / fh);
  const [ox, oy] = [(width - fw * scale) / 2, (height - fh * scale) / 2];
  context.strokeStyle = "#30d158";
  context.lineWidth = 2;
  context.beginPath();
  for (const { quad } of state.lines) {
    quad.forEach(([x, y], i) => (i ? context.lineTo : context.moveTo).call(context, ox + x * scale, oy + y * scale));
    context.closePath();
  }
  context.stroke();
}

async function readFrame() {
  const start = performance.now();
  const shrink = Math.min(1, MAX_FRAME_SIDE / Math.max(video.videoWidth, video.videoHeight));
  frame.width = Math.round(video.videoWidth * shrink);
  frame.height = Math.round(video.videoHeight * shrink);
  frameContext.drawImage(video, 0, 0, frame.width, frame.height);
  const { data } = frameContext.getImageData(0, 0, frame.width, frame.height);
  const frameMs = performance.now() - start;

  const reading = await state.reader.read(data, frame.width, frame.height, state.limit);
  if (state.frozen) return;
  state.text = reading.text;
  state.lines = reading.lines;
  state.frameSize = [frame.width, frame.height];
  $("text").textContent = reading.text;
  $("copy").disabled = !reading.text;
  $("timing").textContent =
    `${reading.lines.length} lines kept of ${reading.detected} found · ${frame.width}×${frame.height} px · ` +
    `frame ${frameMs.toFixed(0)} · detect ${reading.detMs.toFixed(0)} · read ${reading.recMs.toFixed(0)} · ` +
    `total ${(performance.now() - start).toFixed(0)} ms`;
  $("error").textContent = "";
  drawBoxes();
}

/** One job at a time, latest frame only: frames that arrive while busy are skipped. */
function loop() {
  requestAnimationFrame(loop);
  if (state.busy || state.frozen || video.readyState < 2) return;
  state.busy = true;
  readFrame()
    .catch(showError)
    .finally(() => {
      state.busy = false;
    });
}

async function startCamera() {
  $("start").disabled = true; // one camera request at a time
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

$("start").onclick = () =>
  startCamera().catch((error) => {
    $("start").disabled = false;
    showError(error);
  });
$("freeze").onclick = () => {
  state.frozen = !state.frozen;
  $("freeze").textContent = state.frozen ? "Resume" : "Freeze";
  if (state.frozen) video.pause();
  else video.play().catch(showError);
};
$("copy").onclick = () => navigator.clipboard.writeText(state.text).catch(showError);
$("detail").onchange = (event) => {
  state.limit = Number(event.target.value);
};
window.addEventListener("resize", drawBoxes);

async function main() {
  const checks = await probe();
  report(checks, "Loading models…");
  if (!checks.find(([name]) => name === "WebAssembly")[1]) {
    report(checks, "WebAssembly is off, so OCR cannot run here.");
    throw new Error(
      "WebAssembly is blocked. If Lockdown Mode is on: page menu → Website Settings → turn off Lockdown Mode for this site, then reload.",
    );
  }
  const [det, rec, dictionary] = await Promise.all([
    fetchOk("model/det.onnx", "arrayBuffer"),
    fetchOk("model/rec.onnx", "arrayBuffer"),
    fetchOk("model/rec_dict.txt", "text"),
  ]);
  state.reader = await createPageReader(ort, det, rec, dictionary);
  report(checks, await selfTest(state.reader));
  $("start").disabled = !checks.find(([name]) => name.startsWith("Camera"))[1];
}

main().catch(showError);
