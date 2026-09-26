"""Train FlyNet on the reviewed masks (the "student" learning from SAM + human review).

loss = cross-entropy (every pixel's class) + soft Dice on head/thorax/abdomen (overlap per part; stops the
huge background class from dominating). Training uses random dropout masks so the model learns to work
under the same noise that MC dropout applies at inference.

usage: ml/.venv/bin/python ml/train.py [--labels masks] [--epochs 80]
       --labels drafts trains on unreviewed SAM drafts: a plumbing test, never a result.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import FlyParts, splits
from flynet import FlyNet, no_drop, rand_drop

RUNS = Path(__file__).parent / "runs"


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(1)[:, 1:]  # parts only
    t = torch.nn.functional.one_hot(target, 4).permute(0, 3, 1, 2)[:, 1:].float()
    inter, total = (p * t).sum((0, 2, 3)), (p + t).sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (total + 1)).mean()


@torch.no_grad()
def dice_per_part(net, loader, dev) -> list[float]:
    """Dice for head, thorax, abdomen over the whole loader (sums, not an average of per-image scores)."""
    net.eval()
    inter, total = torch.zeros(3), torch.zeros(3)
    for x, y in loader:
        pred = net(x.to(dev), no_drop(len(x), dev)).argmax(1).cpu()
        for k in (1, 2, 3):
            inter[k - 1] += ((pred == k) & (y == k)).sum()
            total[k - 1] += (pred == k).sum() + (y == k).sum()
    return (2 * inter / total.clamp(min=1)).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="masks", choices=["masks", "drafts"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    sp = splits(a.labels)
    print({k: len(v) for k, v in sp.items()}, "images; labels =", a.labels)
    if len(sp["train"]) < 10 or not sp["val"]:
        raise SystemExit("not enough labelled images yet — approve more in the review page")
    train = DataLoader(FlyParts(sp["train"], True, a.labels), batch_size=a.batch, shuffle=True, num_workers=2, persistent_workers=True)
    val = DataLoader(FlyParts(sp["val"], False, a.labels), batch_size=a.batch, num_workers=2)

    out = RUNS / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    net = FlyNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    ce, best, log = torch.nn.CrossEntropyLoss(), -1.0, []

    for ep in range(a.epochs):
        net.train()
        t0, total = time.time(), 0.0
        for x, y in train:
            x, y = x.to(dev), y.to(dev)
            logits = net(x, rand_drop(len(x), device=dev))
            loss = ce(logits, y) + soft_dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        sched.step()
        dice = dice_per_part(net, val, dev)
        mean = sum(dice) / 3
        log.append({"epoch": ep, "loss": total / len(train.dataset), "val_dice": dice, "sec": time.time() - t0})
        print(f"ep {ep:3d} loss {log[-1]['loss']:.3f} val dice head/thorax/abd "
              f"{dice[0]:.3f}/{dice[1]:.3f}/{dice[2]:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if mean > best:
            best = mean
            torch.save({"model": net.state_dict(), "labels": a.labels, "splits": sp, "epoch": ep, "val_dice": dice}, out / "best.pt")
    json.dump(log, open(out / "log.json", "w"), indent=1)
    print(f"best mean val Dice {best:.3f} → {out / 'best.pt'}")


if __name__ == "__main__":
    main()
