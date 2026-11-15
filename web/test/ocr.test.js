// Parity with the model's Python runtime, through the same fixtures the Swift port uses
// (tools/make_fixtures.py). Runs ocr.js on onnxruntime-web's WebAssembly backend in Node.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import * as ort from "onnxruntime-web";
import { PNG } from "pngjs";
import { clean, createRecognizer, isAcceptable } from "../ocr.js";

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
