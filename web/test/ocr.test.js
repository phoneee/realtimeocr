// Parity with the model's Python runtime, through the same fixtures the Swift port uses
// (tools/make_fixtures.py). Runs ocr.js on onnxruntime-web's WebAssembly backend in Node.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import * as ort from "onnxruntime-web";
import { PNG } from "pngjs";
import { clean, createRecognizer, isAcceptable } from "../ocr.js";
import { createPageReader, minAreaRect } from "../page.js";

const root = new URL("../../OCRCore/", import.meta.url);
const fixtures = new URL("Tests/OCRCoreTests/Fixtures/", root);
const model = new URL("Sources/OCRCore/Model/", root);
const expected = JSON.parse(readFileSync(new URL("expected.json", fixtures), "utf8"));

ort.env.wasm.numThreads = 1;

test("reads lines like the reference runtime", async () => {
  const recognizer = await createRecognizer(
    ort,
    readFileSync(new URL("rec.onnx", model)),
    readFileSync(new URL("rec_dict.txt", model), "utf8"),
  );
  for (const line of expected.lines) {
    const png = PNG.sync.read(readFileSync(new URL(line.file, fixtures)));
    const reading = await recognizer.read(png.data, png.width, png.height);
    assert.equal(reading.text, line.text, line.file);
    assert.ok(Math.abs(reading.confidence - line.confidence) < 0.01, `${line.file}: ${reading.confidence}`);
  }
});

// Same lines, near-same text. Boxes agree with the runtime's within 1-2 px, but it rounds
// through integers (cv2 fixed-point resize, pyclipper), and a line's last glyph can read
// differently over 1 px. So: exact line count, and at most 2 % of characters differing.
test("reads pages like the reference runtime", async () => {
  const reader = await createPageReader(
    ort,
    readFileSync(new URL("det.onnx", model)),
    readFileSync(new URL("rec.onnx", model)),
    readFileSync(new URL("rec_dict.txt", model), "utf8"),
  );
  for (const page of expected.pages) {
    const png = PNG.sync.read(readFileSync(new URL(page.file, fixtures)));
    const reading = await reader.read(png.data, png.width, png.height, 1280);
    assert.equal(reading.lines.length, page.lines, page.file);
    const edits = editDistance([...reading.text], [...page.text]);
    assert.ok(edits <= 0.02 * [...page.text].length, `${page.file}: ${edits} edits\n${reading.text}`);
  }
});

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

test("a one-pixel region gives an empty rect instead of failing", () => {
  const rect = minAreaRect([[5, 7], [5, 7]]); // a 1-pixel region: leftmost = rightmost
  assert.equal(Math.min(rect.w, rect.h), 0);
});

test("clean matches reference canonical() and fix_scripts()", () => {
  for (const { input, output } of expected.canonical) {
    assert.equal(clean(input), output, JSON.stringify(input));
  }
});

test("rejects low confidence, short and broken-Thai lines", () => {
  assert.equal(isAcceptable("ราคา", 0.95), true);
  assert.equal(isAcceptable("ราคา", 0.79), false, "under drop_below");
  assert.equal(isAcceptable("ก", 0.99), false, "under min_chars");
  assert.equal(isAcceptable("ูี้ซ์ ริ ร้ร้ ไ้", 0.85), false, "junk Thai spelling");
});
