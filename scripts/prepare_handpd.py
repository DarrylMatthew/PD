"""Convert raw HandPD spiral images into data/handpd/{healthy,parkinson}/<subject>_<n>.png

Source layout: data_raw/handpd/Spiral_HandPD/{SpiralControl,SpiralPatients}/<examID>-<imgID>.jpg
"""
from collections import defaultdict
from pathlib import Path

from PIL import Image

RAW = Path("data_raw/handpd/Spiral_HandPD")
OUT = Path("data/handpd")
SOURCES = {
    "healthy": RAW / "SpiralControl",
    "parkinson": RAW / "SpiralPatients",
}


def main():
    for cls_name, src_dir in SOURCES.items():
        out_dir = OUT / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)

        by_subject = defaultdict(list)
        for p in sorted(src_dir.glob("*.jpg")):
            subject = p.stem.split("-")[0]
            by_subject[subject].append(p)

        n_images = 0
        for subject, paths in by_subject.items():
            for i, p in enumerate(sorted(paths), start=1):
                Image.open(p).convert("RGB").save(out_dir / f"{subject}_{i}.png", "PNG")
                n_images += 1

        print(f"{cls_name}: {len(by_subject)} subjects, {n_images} images -> {out_dir}")


if __name__ == "__main__":
    main()
