"""Train the organ classifier that the counterfactuals have to fool.

This is the model under explanation. It stays small on purpose: the object of
study is the explanation method, and a large classifier would only make the
counterfactual search slower without changing what we measure.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import ORGANS, loader

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"


class OrganCNN(nn.Module):
    """Four-stage CNN over 64x64 slices. Dropout is kept for MC-dropout later."""

    def __init__(self, n_classes=len(ORGANS), width=32, p_drop=0.2):
        super().__init__()
        chans = [1, width, width * 2, width * 4, width * 8]
        blocks = []
        for i in range(4):
            blocks += [
                nn.Conv2d(chans[i], chans[i + 1], 3, padding=1),
                nn.BatchNorm2d(chans[i + 1]),
                nn.SiLU(),
                nn.Conv2d(chans[i + 1], chans[i + 1], 3, padding=1),
                nn.BatchNorm2d(chans[i + 1]),
                nn.SiLU(),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(chans[-1], n_classes)

    def forward(self, x):
        h = self.pool(self.features(x)).flatten(1)
        return self.fc(self.drop(h))


@torch.no_grad()
def evaluate(model, dl, dev):
    """Overall and per-class accuracy on one split."""
    model.eval()
    correct = np.zeros(len(ORGANS))
    total = np.zeros(len(ORGANS))
    for x, y in dl:
        pred = model(x.to(dev)).argmax(1).cpu().numpy()
        y = y.numpy()
        for cls in range(len(ORGANS)):
            m = y == cls
            total[cls] += m.sum()
            correct[cls] += (pred[m] == cls).sum()
    per_class = {ORGANS[i]: float(correct[i] / max(total[i], 1))
                 for i in range(len(ORGANS))}
    return float(correct.sum() / total.sum()), per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="classifier.pt")
    args = ap.parse_args()

    WEIGHTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train = loader("train", args.batch_size)
    val = loader("val", args.batch_size)
    test = loader("test", args.batch_size)

    model = OrganCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_val, history = 0.0, []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y in train:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.shape[0]
            seen += x.shape[0]
        sched.step()
        val_acc, _ = evaluate(model, val, dev)
        history.append({"epoch": epoch + 1, "loss": running / seen, "val_acc": val_acc})
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {running / seen:.4f}"
              f"  val {val_acc:.4f}", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state_dict": model.state_dict()}, WEIGHTS / args.out)

    model.load_state_dict(torch.load(WEIGHTS / args.out)["state_dict"])
    test_acc, per_class = evaluate(model, test, dev)
    print(f"test accuracy {test_acc:.4f}")

    (RESULTS / "classifier.json").write_text(json.dumps(
        {"epochs": args.epochs, "val_acc": best_val, "test_acc": test_acc,
         "per_class_test_acc": per_class, "minutes": (time.time() - start) / 60,
         "history": history}, indent=2))


if __name__ == "__main__":
    main()
