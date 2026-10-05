"""Write a normalized copy of data/ to data_norm/ (same layout and filenames).

Per image: resize to 256x256, divide out the estimated paper background (removes
tint and lighting), then stretch contrast so faint pencil and strong pen look alike.
Raw data/ is left untouched.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SRC = Path("data")
DST = Path("data_norm")
SIZE = 256
BG_KERNEL = 21
INK_PERCENTILE = 99.0


def normalize_ink(img):
    g = img.convert("L").resize((SIZE, SIZE), Image.BILINEAR)
    bg = g.filter(ImageFilter.MaxFilter(BG_KERNEL)).filter(ImageFilter.GaussianBlur(BG_KERNEL / 2))
    a = np.asarray(g, dtype=np.float32)
    b = np.maximum(np.asarray(bg, dtype=np.float32), 1.0)
    ink = np.clip(1.0 - a / b, 0.0, 1.0)
    hi = np.percentile(ink, INK_PERCENTILE)
    if hi > 1e-3:
        ink = np.clip(ink / hi, 0.0, 1.0)
    return Image.fromarray(((1.0 - ink) * 255).astype(np.uint8))


def main():
    for src in sorted(SRC.glob("*/*/*.png")):
        dst = DST / src.relative_to(SRC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        normalize_ink(Image.open(src)).save(dst, "PNG")
    for ds in sorted(p for p in DST.iterdir() if p.is_dir()):
        counts = {c.name: len(list(c.glob("*.png"))) for c in sorted(ds.iterdir())}
        print(ds.name, counts)


if __name__ == "__main__":
    main()
