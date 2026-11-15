"""Render Thai/English line and page images and record what the official runtime reads.

The Swift and JS ports are checked against these: same pixels in, same text out.

Usage:
    python -I tools/make_fixtures.py <hf_snapshot_dir> <out_dir>

<hf_snapshot_dir> holds det.onnx, det.yml, rec.onnx, rec_dict.txt, config.json and
runtime/ from https://huggingface.co/iapp/OpenThai2.0-OCR-Text.
Needs: onnxruntime opencv-python-headless pyclipper numpy pillow.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Supplemental/Sathu.ttf"
HEIGHT = 96  # taller than the recognizer's 48 px, so the resize path is exercised
FONT_SIZE = 52
PAD_X = 24

# (name, text, ink, paper): one dark-on-light, one light-on-dark (flatten must flip it).
LINES = [
    ("price", "ราคา 1,250 บาท รวมภาษีมูลค่าเพิ่มแล้ว", 20, 245),
    ("address", "กรุงเทพมหานคร Bangkok 10110", 30, 235),
    ("water", "น้ำดื่มสะอาด ผ่านการกรองหลายขั้นตอน", 240, 40),
]

# Whole pages for detector + reading order. One column, no lone tall characters:
# the ports leave out column splitting and reading tall boxes turned.
PAGE_LINES = [
    "ประกาศ เรื่อง การปิดปรับปรุงระบบ",
    "วันเสาร์ที่ 21 พฤศจิกายน 2569 เวลา 22.00 น.",
    "ถึงวันอาทิตย์ที่ 22 พฤศจิกายน 2569 เวลา 06.00 น.",
    "ขออภัยในความไม่สะดวก Thank you",
    "โทร 02-123-4567 อีเมล support@example.com",
]
PAGES = [("page_straight", 0.0), ("page_tilted", 2.0)]  # degrees; under the 2.5 deskew threshold
PAGE_SIZE = (1200, 640)
PAGE_FONT_SIZE = 40
PAGE_MARGIN = 60
PAGE_LEADING = 100

# Parts of LineOCR.read() the ports do not implement, switched off for the reference run.
PAGE_OVERRIDES = {"ori": {"enabled": False}, "det": {"split_gap": 0}, "rec": {"batch": 1}}

# canonical() cases: input -> what the reference returns.
CANONICAL_INPUTS = [
    "นํ้า",            # nikhahit + tone + aa -> tone + sara am
    "น้ํา",            # tone + nikhahit + aa
    "เเม่",            # two sara e -> sara ae
    "กี่ ่",           # stray mark after space stays
    "ก่ิ",             # tone before vowel -> vowel first
    "ก​ข  ค",  # zero-width space dropped, NBSP -> space, spaces collapsed
    "ก",         # legacy PUA tone mark -> standard
    "Wฤษภาคม",         # lone W before Thai -> พ (fix_scripts)
    "๒O๕",             # Latin O between Thai digits -> ๐ (fix_scripts)
]


def render_line(text: str, ink: int, paper: int) -> Image.Image:
    font = ImageFont.truetype(FONT, FONT_SIZE)
    left, _, right, _ = font.getbbox(text)
    img = Image.new("RGB", (right - left + 2 * PAD_X, HEIGHT), (paper,) * 3)
    ImageDraw.Draw(img).text((PAD_X - left, HEIGHT // 2), text, font=font, fill=(ink,) * 3, anchor="lm")
    return img


def render_page(angle: float) -> Image.Image:
    font = ImageFont.truetype(FONT, PAGE_FONT_SIZE)
    img = Image.new("RGB", PAGE_SIZE, (242, 240, 235))
    draw = ImageDraw.Draw(img)
    for n, text in enumerate(PAGE_LINES):
        draw.text((PAGE_MARGIN, PAGE_MARGIN + n * PAGE_LEADING), text, font=font, fill=(25, 25, 30))
    return img.rotate(angle, resample=Image.BICUBIC, fillcolor=(242, 240, 235)) if angle else img


def main(hf_dir: Path, out_dir: Path) -> None:
    sys.path.insert(0, str(hf_dir))
    import cv2  # noqa: E402
    import numpy as np  # noqa: E402
    from runtime.line_engine import LineOCR, fix_scripts  # noqa: E402
    from runtime.thai_norm import canonical  # noqa: E402

    def bgr(path: Path) -> np.ndarray:
        return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)

    out_dir.mkdir(parents=True, exist_ok=True)
    ocr = LineOCR(str(hf_dir), threads=1)
    lines = []
    for name, truth, ink, paper in LINES:
        path = out_dir / f"{name}.png"
        render_line(truth, ink, paper).save(path)
        (raw, conf), = ocr.recognize([bgr(path)])
        lines.append({"file": path.name, "truth": truth,
                      "text": fix_scripts(canonical(raw)), "confidence": round(conf, 4)})

    page_ocr = LineOCR(str(hf_dir), threads=1, **PAGE_OVERRIDES)
    pages = []
    for name, angle in PAGES:
        path = out_dir / f"{name}.png"
        render_page(angle).save(path)
        result = page_ocr.read(bgr(path))
        pages.append({"file": path.name, "truth": "\n".join(PAGE_LINES), "text": result["text"],
                      "lines": len(result["lines"]), "det": page_ocr.cfg["det"]})

    cases = [{"input": s, "output": fix_scripts(canonical(s))} for s in CANONICAL_INPUTS]
    (out_dir / "expected.json").write_text(
        json.dumps({"lines": lines, "pages": pages, "canonical": cases}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    for item in lines + pages:
        print(item)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
