# Realtime Thai OCR prototype

Point the camera at one printed line, keep it inside the yellow box, and read it live.
Everything runs on the device: frames never leave it.

Model: [iapp/OpenThai2.0-OCR-Text](https://huggingface.co/iapp/OpenThai2.0-OCR-Text), the
recognizer only (`rec.onnx` + `rec_dict.txt`, Apache-2.0, see `OCRCore/Sources/OCRCore/Model/NOTICE`).
The line is chosen by the guide box, so the detector is not used yet.

Both versions port the model runtime's recognizer path (resize to 48 px, `_flatten`,
greedy CTC, `canonical` + `fix_scripts`) and are tested against what the Python runtime
reads on the same crops (`tools/make_fixtures.py`).

## iOS (`App/`, `OCRCore/`)

SwiftUI + AVFoundation + ONNX Runtime (CPU).

```sh
(cd OCRCore && swift test)   # parity tests, run on the Mac
xcodegen generate            # writes RealtimeOCR.xcodeproj from project.yml
```

Set `DEVELOPMENT_TEAM` in `project.yml` to your team, then build and run on a device from Xcode.
The device needs Developer Mode on.

## Web (`web/`)

ONNX Runtime Web (WebAssembly, single thread). Deployed to GitHub Pages by `.github/workflows/pages.yml`.

```sh
cd web
npm ci
npm test                     # parity tests in Node
npm run build                # assembles dist/
python3 -m http.server -d dist 8000   # open http://localhost:8000
```

On load the page lists what the browser allows (camera, WebAssembly, SIMD) and reads a
reference line as a self-test. With iOS Lockdown Mode on, WebAssembly and the camera API
are off: exclude this site in Safari (page menu → Website Settings → Lockdown Mode off).

## Regenerating fixtures

```sh
python -I tools/make_fixtures.py <hf_snapshot_dir> OCRCore/Tests/OCRCoreTests/Fixtures
```
