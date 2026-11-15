"""Render Thai/English line images and record what the official runtime reads.

The Swift port is checked against these: same crops in, same text out.

Usage:
    python -I tools/make_fixtures.py <hf_snapshot_dir> <out_dir>

<hf_snapshot_dir> holds det.onnx, rec.onnx, rec_dict.txt, config.json and
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


def render(text: str, ink: int, paper: int) -> Image.Image:
    font = ImageFont.truetype(FONT, FONT_SIZE)
    left, _, right, _ = font.getbbox(text)
    img = Image.new("RGB", (right - left + 2 * PAD_X, HEIGHT), (paper,) * 3)
    ImageDraw.Draw(img).text((PAD_X - left, HEIGHT // 2), text, font=font, fill=(ink,) * 3, anchor="lm")
    return img


def main(hf_dir: Path, out_dir: Path) -> None:
    sys.path.insert(0, str(hf_dir))
    import cv2  # noqa: E402
    import numpy as np  # noqa: E402
    from runtime.line_engine import LineOCR, fix_scripts  # noqa: E402
    from runtime.thai_norm import canonical  # noqa: E402

    ocr = LineOCR(str(hf_dir), threads=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, truth, ink, paper in LINES:
        path = out_dir / f"{name}.png"
        render(truth, ink, paper).save(path)
        bgr = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        (raw, conf), = ocr.recognize([bgr])
        lines.append({"file": path.name, "truth": truth,
                      "text": fix_scripts(canonical(raw)), "confidence": round(conf, 4)})
    cases = [{"input": s, "output": fix_scripts(canonical(s))} for s in CANONICAL_INPUTS]
    (out_dir / "expected.json").write_text(
        json.dumps({"lines": lines, "canonical": cases}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    for case in cases:
        print(case)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
