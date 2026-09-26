"""Draft head / thorax / abdomen masks with Segment Anything, for a human to approve or fix.

SAM is a "teacher": it segments almost anything well but needs prompts (points) and is far too big for a
browser (~375 MB). We prompt it automatically, a human reviews the drafts, and the approved masks train a
small "student" model (train.py) — that is knowledge distillation from a foundation model.

Automatic prompts, per fly:
  1. fly pixels = saturated (yellow/orange body) and not dark; an opening strips legs and wings
  2. body axis = principal axis of those pixels (second moments)
  3. head end = the end nearer the red/orange eye pixels (fallback: the narrower end)
  4. one point each at 8% / 35% / 72% along the axis → head / thorax / abdomen; the other two parts'
     points are negative prompts, so SAM separates the parts instead of returning the whole fly.

Drafts are DRAFTS. Only masks approved in the review page (ml/data/masks) are ground truth.

usage: ml/.venv/bin/python ml/sam_draft.py [--per-session 5] [--cap 150]
"""
import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

DATA = Path(__file__).parent / "data"
PARTS = {1: ("head", 0.08, 0.10), 2: ("thorax", 0.35, 0.28), 3: ("abdomen", 0.72, 0.45)}  # label: name, axis pos, expected area share
COLORS = {1: (60, 60, 255), 2: (255, 120, 40), 3: (80, 220, 80)}  # BGR: head red, thorax blue, abdomen green


class Sam:
    def __init__(self, device: str | None = None):
        self.dev = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.dev).eval()
        self.proc = SamProcessor.from_pretrained("facebook/sam-vit-base")

    @torch.no_grad()
    def embed(self, rgb: np.ndarray):
        inp = self.proc(Image.fromarray(rgb), return_tensors="pt")
        return self.model.get_image_embeddings(inp["pixel_values"].to(self.dev)), inp

    @torch.no_grad()
    def masks(self, emb, inp, points: list[list[tuple[float, float]]], labels: list[list[int]]):
        """points[m] = prompts for mask m (all masks need the same number of points).
        Returns masks [M, 3, H, W] (bool) and IoU scores [M, 3]."""
        s = inp["reshaped_input_sizes"][0, 0].item() / inp["original_sizes"][0, 0].item()
        pts = torch.tensor([points], dtype=torch.float32) * s
        out = self.model(image_embeddings=emb, input_points=pts.to(self.dev),
                         input_labels=torch.tensor([labels]).to(self.dev), multimask_output=True)
        m = self.proc.image_processor.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"], inp["reshaped_input_sizes"])[0]
        return m.numpy().astype(bool), out.iou_scores[0].cpu().numpy()


def fly_blobs(bgr: np.ndarray) -> list[np.ndarray]:
    """Boolean masks of each fly body (legs/wings mostly removed), largest first."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = cv2.GaussianBlur(hsv[:, :, 1], (7, 7), 0)
    t, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = ((s > max(t, 60)) & (hsv[:, :, 2] > 50)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.015 * fg.size]
    keep.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    return [lab == i for i in keep[:3]]


def axis_points(bgr: np.ndarray, blob: np.ndarray, flip: bool = False) -> dict[int, tuple[float, float]]:
    ys, xs = np.nonzero(blob)
    c = np.array([xs.mean(), ys.mean()])
    evals, evecs = np.linalg.eigh(np.cov(np.stack([xs, ys])))
    u = evecs[:, np.argmax(evals)]  # long axis direction
    t = (np.stack([xs, ys], 1) - c) @ u
    lo, hi = np.percentile(t, 1), np.percentile(t, 99)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    near = cv2.dilate(blob.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    # Eyes are redder than this fly's own body (red or orange eyes on a yellow-tan body), so compare to the body
    body_h, body_s = np.median(hsv[:, :, 0][blob]), np.median(hsv[:, :, 1][blob])
    h = hsv[:, :, 0].astype(int)
    eye = near & ((h < body_h - 6) | (h > 165)) & (hsv[:, :, 1] > body_s) & (hsv[:, :, 2] > 60)
    if eye.sum() > 50:
        ey, ex = np.nonzero(eye)
        head_is_lo = ((np.stack([ex, ey], 1) - c) @ u).mean() < (lo + hi) / 2
    else:  # no eye colour: the head end is the narrower end
        perp = (np.stack([xs, ys], 1) - c) @ np.array([-u[1], u[0]])
        w_lo = np.ptp(perp[t < lo + 0.15 * (hi - lo)]) if (t < lo + 0.15 * (hi - lo)).any() else 0
        w_hi = np.ptp(perp[t > hi - 0.15 * (hi - lo)]) if (t > hi - 0.15 * (hi - lo)).any() else 0
        head_is_lo = w_lo < w_hi
    start, end = (lo, hi) if head_is_lo != flip else (hi, lo)
    return {k: tuple(c + u * (start + f * (end - start))) for k, (_, f, _) in PARTS.items()}


def draft(sam: Sam, bgr: np.ndarray, flip: bool = False) -> np.ndarray:
    """Label map (0 bg, 1 head, 2 thorax, 3 abdomen) for every fly in the image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    emb, inp = sam.embed(rgb)
    out = np.zeros(bgr.shape[:2], np.uint8)
    for blob in fly_blobs(bgr):
        pts = axis_points(bgr, blob, flip)
        order = list(PARTS)
        prompts = [[pts[k]] + [pts[o] for o in order if o != k] for k in order]
        labels = [[1, 0, 0] for _ in order]
        masks, scores = sam.masks(emb, inp, prompts, labels)
        region = cv2.dilate(blob.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        body_area = blob.sum()
        chosen = {}
        for i, k in enumerate(order):
            share = PARTS[k][2]
            ok = [j for j in range(3) if scores[i, j] >= 0.7 * scores[i].max()]
            j = min(ok, key=lambda j: abs(np.log(((masks[i, j] & region).sum() + 1) / (share * body_area))))
            chosen[k] = masks[i, j] & region
        for k in sorted(chosen, key=lambda k: -chosen[k].sum()):  # paint big first, so smaller parts win overlaps
            out[chosen[k]] = k
    return out


def preview(bgr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    over = bgr.copy()
    for k, col in COLORS.items():
        over[labels == k] = (0.5 * over[labels == k] + 0.5 * np.array(col)).astype(np.uint8)
    return over


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-session", type=int, default=5)
    ap.add_argument("--cap", type=int, default=150)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(DATA / "manifest.csv")))
    random.seed(0)
    by_session: dict[str, list[dict]] = {}
    for r in rows:
        by_session.setdefault(r["session"], []).append(r)
    pick = [r for rs in by_session.values() for r in random.sample(rs, min(a.per_session, len(rs)))][: a.cap]

    (DATA / "drafts").mkdir(exist_ok=True)
    (DATA / "previews").mkdir(exist_ok=True)
    sam, t0, done = Sam(), time.time(), 0
    for r in pick:
        out = DATA / "drafts" / f"{r['sha1']}.png"
        if out.exists():
            continue
        bgr = cv2.imread(str(DATA / "img" / f"{r['sha1']}.jpg"))
        labels = draft(sam, bgr)
        cv2.imwrite(str(out), labels)
        cv2.imwrite(str(DATA / "previews" / f"{r['sha1']}.jpg"), preview(bgr, labels))
        done += 1
        print(f"{done}/{len(pick)} {r['sha1'][:8]} parts={sorted(set(np.unique(labels)) - {0})} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
