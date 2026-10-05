"""Convert raw NewHandPD spiral images into data/newhandpd/{healthy,parkinson}/<subject>_<n>.png

Source layout: data_raw/newhandpd/{HealthySpiral/HealthySpiral,PatientSpiral/PatientSpiral}/sp<n>-<subject>.jpg
"""
from collections import defaultdict
from pathlib import Path

from PIL import Image

RAW = Path("data_raw/newhandpd")
OUT = Path("data/newhandpd")
SOURCES = {
    "healthy": RAW / "HealthySpiral" / "HealthySpiral",
    "parkinson": RAW / "PatientSpiral" / "PatientSpiral",
}


def main():
    for cls_name, src_dir in SOURCES.items():
        out_dir = OUT / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)

        by_subject = defaultdict(list)
        for p in sorted(src_dir.glob("*.jpg")):
            # filename is sp<n>-<subject>.jpg, e.g. sp1-H1.jpg -> subject H1
            subject = p.stem.split("-", 1)[1]
            by_subject[subject].append(p)

        n_images = 0
        for subject, paths in by_subject.items():
            for i, p in enumerate(sorted(paths), start=1):
                Image.open(p).convert("RGB").save(out_dir / f"{subject}_{i}.png", "PNG")
                n_images += 1

        print(f"{cls_name}: {len(by_subject)} subjects, {n_images} images -> {out_dir}")


if __name__ == "__main__":
    main()
