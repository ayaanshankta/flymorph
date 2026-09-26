"""Score a trained FlyNet on its held-out TEST sessions, the way the app uses it (8 MC-dropout passes).

Reports, per part (head / thorax / abdomen / body):
  dice        pixel overlap with the label masks (1 = perfect)
  area_err    median |predicted − labelled| area per fly, as % of the labelled area — the number that matters
  coverage    how often the labelled area falls inside predicted mean ± 2 SD (≈ 95% if the error bars are honest)

A test Dice above 0.98 is treated as a leak until disproven. Metrics from --labels drafts compare the model
with SAM's unreviewed drafts: that shows the pipeline runs, not how accurate it is.

usage: ml/.venv/bin/python ml/evaluate.py ml/runs/<run>/best.pt
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from data import DATA, FlyParts
from flynet import FlyNet, rand_drop

T = 8
PARTS = {"head": [1], "thorax": [2], "abdomen": [3], "body": [1, 2, 3]}


@torch.no_grad()
def main(ckpt_path: str) -> None:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = FlyNet(pretrained=False).to(dev).eval()
    net.load_state_dict(ck["model"])
    ds = FlyParts(ck["splits"]["test"], False, ck["labels"])
    session = {r["sha1"]: r["session"] for r in csv.DictReader(open(DATA / "manifest.csv"))}

    inter = {p: 0 for p in PARTS}
    tot = {p: 0 for p in PARTS}
    errs = {p: [] for p in PARTS}
    covered = {p: [] for p in PARTS}
    for i in range(len(ds)):
        x, y = ds[i]
        probs = net(x[None].repeat(T, 1, 1, 1).to(dev), rand_drop(T, device=dev)).softmax(1).cpu()
        per_pass, mean = probs.argmax(1).numpy(), probs.mean(0).argmax(0).numpy()
        y = y.numpy()
        for p, ks in PARTS.items():
            pm, ym = np.isin(mean, ks), np.isin(y, ks)
            inter[p] += (pm & ym).sum()
            tot[p] += pm.sum() + ym.sum()
        # per fly: labelled flies = connected blobs of labelled body; compare areas inside each (dilated) region
        n, flies = cv2.connectedComponents((y > 0).astype(np.uint8))
        for f in range(1, n):
            region = cv2.dilate((flies == f).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
            for p, ks in PARTS.items():
                truth = (np.isin(y, ks) & (flies == f)).sum()
                if truth < 50:
                    continue
                areas = np.array([(np.isin(per_pass[t], ks) & region).sum() for t in range(T)])
                errs[p].append(abs(areas.mean() - truth) / truth * 100)
                covered[p].append(abs(areas.mean() - truth) <= 2 * areas.std() + 1)

    metrics = {
        "labels": ck["labels"],
        "ground_truth": ck["labels"] == "masks",
        "test_images": len(ds),
        "test_sessions": len({session[s] for s in ck["splits"]["test"]}),
        "parts": {p: {"dice": round(2 * inter[p] / max(tot[p], 1), 4),
                      "median_area_err_pct": round(float(np.median(errs[p])), 2) if errs[p] else None,
                      "coverage_2sd": round(float(np.mean(covered[p])), 3) if covered[p] else None,
                      "flies": len(errs[p])} for p in PARTS},
    }
    out = Path(ckpt_path).parent / "metrics.json"
    json.dump(metrics, open(out, "w"), indent=1)
    print(json.dumps(metrics, indent=1))
    if not metrics["ground_truth"]:
        print("NOTE: scored against unreviewed SAM drafts — pipeline check only, not accuracy.")
    if any(v["dice"] > 0.98 for v in metrics["parts"].values()):
        print("WARNING: Dice > 0.98 — check for train/test leakage before believing this.")


if __name__ == "__main__":
    main(sys.argv[1])
