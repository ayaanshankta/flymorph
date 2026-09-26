"""Training pairs = approved masks (ml/data/masks) + their cached photos, split by imaging session.

Why by session: photos from one session are often the same flies moved slightly. If one lands in train and
its sibling in test, the test score measures memory, not skill. Sessions come from manifest.py.
"""
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from flynet import SIZE

DATA = Path(__file__).parent / "data"
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def splits(labels: str = "masks", seed: int = 0, frac=(0.7, 0.15, 0.15)) -> dict[str, list[str]]:
    """{'train'|'val'|'test': [sha1, ...]} over label PNGs in ml/data/<labels>, whole sessions per split.
    labels="masks" = human-approved ground truth; "drafts" = raw SAM output, only for plumbing tests."""
    session = {r["sha1"]: r["session"] for r in csv.DictReader(open(DATA / "manifest.csv"))}
    shas = sorted(p.stem for p in (DATA / labels).glob("*.png"))
    sessions = sorted({session[s] for s in shas})
    random.Random(seed).shuffle(sessions)
    n = len(sessions)
    cut1, cut2 = round(frac[0] * n), round((frac[0] + frac[1]) * n)
    groups = {"train": set(sessions[:cut1]), "val": set(sessions[cut1:cut2]), "test": set(sessions[cut2:])}
    out = {k: [s for s in shas if session[s] in g] for k, g in groups.items()}
    # the file names ARE content hashes, so disjoint names = no photo in two splits
    assert not (set(out["train"]) & set(out["test"])) and not (set(out["train"]) & set(out["val"]))
    return out


def letterbox(img: np.ndarray, interp: int) -> np.ndarray:
    """Same geometry as core/adult.ts preprocess: long side → SIZE, centred, zero padding."""
    h, w = img.shape[:2]
    s = SIZE / max(h, w)
    nw, nh = round(w * s), round(h * s)
    out = np.zeros((SIZE, SIZE) + img.shape[2:], img.dtype)
    x0, y0 = (SIZE - nw) // 2, (SIZE - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return out


class FlyParts(Dataset):
    def __init__(self, shas: list[str], augment: bool, labels: str = "masks"):
        self.shas, self.augment, self.labels = shas, augment, labels

    def __len__(self):
        return len(self.shas)

    def __getitem__(self, i):
        sha = self.shas[i]
        img = cv2.cvtColor(cv2.imread(str(DATA / "img" / f"{sha}.jpg")), cv2.COLOR_BGR2RGB)
        lab = cv2.imread(str(DATA / self.labels / f"{sha}.png"), cv2.IMREAD_GRAYSCALE)
        if self.augment:
            img, lab = augment(img, lab)
        img = letterbox(img, cv2.INTER_AREA)
        lab = letterbox(lab, cv2.INTER_NEAREST)  # nearest: never invent a label between two classes
        x = torch.from_numpy(((img / 255.0 - MEAN) / STD).transpose(2, 0, 1)).float()
        return x, torch.from_numpy(lab).long()


def augment(img: np.ndarray, lab: np.ndarray):
    """Random flips, 90° turns, small rotation and colour jitter — flies lie in every orientation and
    lighting differs per session, so the model must not rely on either."""
    if random.random() < 0.5:
        img, lab = img[:, ::-1], lab[:, ::-1]
    if random.random() < 0.5:
        img, lab = img[::-1], lab[::-1]
    k = random.randint(0, 3)
    img, lab = np.rot90(img, k), np.rot90(lab, k)
    img, lab = np.ascontiguousarray(img), np.ascontiguousarray(lab)
    h, w = lab.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-20, 20), 1.0)
    img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    lab = cv2.warpAffine(lab, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-6, 6)) % 180
    hsv[..., 1] *= random.uniform(0.7, 1.3)
    hsv[..., 2] *= random.uniform(0.7, 1.3)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img, lab
