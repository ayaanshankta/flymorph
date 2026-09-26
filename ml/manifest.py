"""Index every adult-fly photo once: content hash, imaging session, and a small cached copy.

Originals are never modified or copied at full size. Output (git-ignored):
  ml/data/manifest.csv       sha1, session, path, width, height
  ml/data/img/<sha1>.jpg     long side 1024 px — everything downstream reads these

Session = nearest folder that holds a scale image: everything calibrated by the same scale was shot
together, and the train/val/test split must keep a session on one side (otherwise near-identical
photos of the same fly leak into the test set).

usage: ml/.venv/bin/python ml/manifest.py [--root ~/Downloads/FlyAreaDetect]
"""
import argparse
import csv
import hashlib
from pathlib import Path

import cv2

LONG_SIDE = 1024
OUT = Path(__file__).parent / "data"
SOURCES = ["data/adult", "ML-data/Microscope Images"]


def session_of(img: Path, root: Path) -> str:
    for d in img.parents:
        if d == root:
            break
        if any("scale" in p.name.lower() for p in d.glob("*.jp*g")):
            return str(d.relative_to(root))
    d = img.parent
    if d.name.lower() in {"top", "side"}:  # two views of the same flies = one session
        d = d.parent
    return str(d.relative_to(root))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.home() / "Downloads/FlyAreaDetect")
    root = ap.parse_args().root
    (OUT / "img").mkdir(parents=True, exist_ok=True)

    rows, seen = [], set()
    for src in SOURCES:
        for p in sorted((root / src).rglob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"} or "scale" in p.name.lower():
                continue
            sha = hashlib.sha1(p.read_bytes()).hexdigest()
            if sha in seen:  # the same photo saved twice would otherwise sit in train AND test
                continue
            seen.add(sha)
            im = cv2.imread(str(p))
            if im is None:
                print(f"unreadable, skipped: {p}")
                continue
            h, w = im.shape[:2]
            cache = OUT / "img" / f"{sha}.jpg"
            if not cache.exists():
                s = LONG_SIDE / max(h, w)
                cv2.imwrite(str(cache), cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append({"sha1": sha, "session": session_of(p, root), "path": str(p.relative_to(root)), "width": w, "height": h})

    with open(OUT / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["sha1", "session", "path", "width", "height"])
        wr.writeheader()
        wr.writerows(rows)
    sessions = {r["session"] for r in rows}
    print(f"{len(rows)} unique images in {len(sessions)} sessions")


if __name__ == "__main__":
    main()
