# Realtime Thai OCR prototype

Point the camera at printed text and read everything in view, live, with a box on each line found.
Everything runs on the device: frames never leave it.

Model: [iapp/OpenThai2.0-OCR-Text](https://huggingface.co/iapp/OpenThai2.0-OCR-Text)
(`det.onnx`, `rec.onnx`, `rec_dict.txt`; Apache-2.0, see `OCRCore/Sources/OCRCore/Model/NOTICE`).

Both versions port the model runtime's page path: detector with DB post-processing, each line
cut out (perspective, bicubic) and read (48 px, `_flatten`, greedy CTC, `canonical` +
`fix_scripts`), rows put in reading order, fragments and low-confidence lines dropped. They are
tested against what the Python runtime reads on the same images (`tools/make_fixtures.py`):
line crops exactly, whole pages to the same line count and within 2 % of characters (the
runtime rounds through integers in places, which moves boxes by under a pixel).

Not ported yet: page orientation (sideways / upside down), deskew-and-redetect, splitting lines
at wide gaps, multi-column reading order, and reading tall boxes turned. A phone held upright
over one column of text does not need them.

The Detail setting is the detector's long side (736 / 960 / 1280 px): faster against smaller text.

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
reference page as a self-test. With iOS Lockdown Mode on, WebAssembly and the camera API
are off: exclude this site in Safari (page menu → Website Settings → Lockdown Mode off).

## Regenerating fixtures

```sh
python -I tools/make_fixtures.py <hf_snapshot_dir> OCRCore/Tests/OCRCoreTests/Fixtures
```
