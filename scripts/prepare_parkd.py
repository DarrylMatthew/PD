"""Convert raw ParkD (Kaggle, Zham et al.) spiral images into data/parkd/{healthy,parkinson}/<subject>_<n>.png

Source layout: data_raw/parkd/archive (1)/spiral/{training,testing}/{healthy,parkinson}/V<num><H|P>E<exam>.png
The Kaggle training/testing split is ignored here; baseline.py does its own subject-wise split.
The 'drawings/' and 'wave/' subfolders are ignored (drawings/spiral is a duplicate of spiral/, wave/ is not used).
"""
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

RAW = Path("data_raw/parkd/archive (1)/spiral")
OUT = Path("data/parkd")
SOURCES = {
    "healthy": [RAW / "training" / "healthy", RAW / "testing" / "healthy"],
    "parkinson": [RAW / "training" / "parkinson", RAW / "testing" / "parkinson"],
}
# V01H and V01P are different people (the number resets per class), so the
# subject id must include the H/P letter.
SUBJECT_RE = re.compile(r"^(V\d+[HP])")


def main():
    for cls_name, src_dirs in SOURCES.items():
        out_dir = OUT / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)

        by_subject = defaultdict(list)
        for src_dir in src_dirs:
            for p in sorted(src_dir.glob("*.png")):
                m = SUBJECT_RE.match(p.stem)
                if not m:
                    raise ValueError(f"Unexpected filename pattern: {p}")
                by_subject[m.group(1)].append(p)

        n_images = 0
        for subject, paths in by_subject.items():
            for i, p in enumerate(sorted(paths), start=1):
                Image.open(p).convert("RGB").save(out_dir / f"{subject}_{i}.png", "PNG")
                n_images += 1

        print(f"{cls_name}: {len(by_subject)} subjects, {n_images} images -> {out_dir}")


if __name__ == "__main__":
    main()
