"""Build every figure used in the README from the saved result files."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import ORGANS, class_counts

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_class_distribution():
    counts = class_counts("train")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(list(counts.keys()), list(counts.values()), color="#3b6ea5")
    ax.set_ylabel("training slices")
    ax.set_title("OrganAMNIST class distribution, train split")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGS / "class_distribution.png")
    plt.close()


def fig_training_curves():
    dif = load("diffusion_training.json")
    clf = load("classifier.json")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    if dif:
        h = dif["history"]
        axes[0].plot([e["epoch"] for e in h], [e["loss"] for e in h], color="#3b6ea5")
        axes[0].set_title("DDPM noise-prediction loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("MSE")
    if clf:
        h = clf["history"]
        axes[1].plot([e["epoch"] for e in h], [e["val_acc"] for e in h], color="#b5651d")
        axes[1].set_title("Classifier validation accuracy")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("accuracy")
    plt.tight_layout()
    plt.savefig(FIGS / "training_curves.png")
    plt.close()


def fig_samples():
    p = RESULTS / "samples.pt"
    if not p.exists():
        return
    s = torch.load(p)[:32].squeeze(1).numpy()
    fig, axes = plt.subplots(4, 8, figsize=(8, 4.2))
    for ax, img in zip(axes.flat, s):
        ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
    fig.suptitle("Unconditional DDPM samples")
    plt.tight_layout()
    plt.savefig(FIGS / "samples.png")
    plt.close()


def _best_operating_point():
    """start_t with the highest validity, which is what the panel should show."""
    d = load("counterfactual_sweep.json")
    if not d:
        return None
    return max(d["sweep"], key=lambda r: r["validity"])["start_t"]


def fig_counterfactual_panel():
    t = _best_operating_point()
    p = RESULTS / f"counterfactual_examples_t{t}.pt" if t else None
    if p is None or not p.exists():
        cands = sorted(RESULTS.glob("counterfactual_examples_t*.pt"))
        if not cands:
            return
        p = cands[0]
    d = torch.load(p)
    x0, xcf = d["x0"].squeeze(1).numpy(), d["xcf"].squeeze(1).numpy()
    src, tgt = d["source"].numpy(), d["target"].numpy()
    n = min(8, len(x0))
    fig, axes = plt.subplots(3, n, figsize=(1.35 * n, 4.6))
    for j in range(n):
        axes[0, j].imshow(x0[j], cmap="gray", vmin=-1, vmax=1)
        axes[0, j].set_title(ORGANS[src[j]], fontsize=7)
        axes[1, j].imshow(xcf[j], cmap="gray", vmin=-1, vmax=1)
        axes[1, j].set_title(f"to {ORGANS[tgt[j]]}", fontsize=7)
        diff = xcf[j] - x0[j]
        lim = max(abs(diff).max(), 1e-6)
        axes[2, j].imshow(diff, cmap="bwr", vmin=-lim, vmax=lim)
        for r in range(3):
            axes[r, j].axis("off")
    for r, lab in enumerate(["original", "counterfactual", "difference"]):
        axes[r, 0].set_ylabel(lab)
    fig.suptitle(f"Counterfactuals at start_t = {d['start_t']}")
    plt.tight_layout()
    plt.savefig(FIGS / "counterfactual_panel.png")
    plt.close()


def fig_validity_proximity():
    d = load("counterfactual_sweep.json")
    if not d:
        return
    s = d["sweep"]
    t = [r["start_t"] for r in s]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].plot(t, [r["validity"] for r in s], "o-", color="#3b6ea5", label="reaches target")
    axes[0].plot(t, [r["flip_rate_any"] for r in s], "s--", color="#8a8a8a", label="differs from dataset label")
    axes[0].set_xlabel("start_t")
    axes[0].set_ylabel("rate")
    axes[0].set_title("Validity")
    axes[0].legend(fontsize=7)
    axes[1].plot(t, [r["l1"] for r in s], "o-", color="#b5651d")
    axes[1].set_xlabel("start_t")
    axes[1].set_ylabel("mean L1")
    axes[1].set_title("Proximity, lower is closer")
    axes[2].plot([r["l1"] for r in s], [r["validity"] for r in s], "o-", color="#2e7d5b")
    for r in s:
        axes[2].annotate(str(r["start_t"]), (r["l1"], r["validity"]), fontsize=6,
                         xytext=(3, 3), textcoords="offset points")
    axes[2].set_xlabel("mean L1 change")
    axes[2].set_ylabel("validity")
    axes[2].set_title("The trade-off")
    plt.tight_layout()
    plt.savefig(FIGS / "validity_proximity.png")
    plt.close()


def fig_memorisation():
    d = load("sample_quality.json")
    if not d:
        return
    m = d["memorisation"]
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = ["generated", "real test"]
    med = [m["generated_nn_distance"]["median"], m["real_test_nn_distance"]["median"]]
    p5 = [m["generated_nn_distance"]["p5"], m["real_test_nn_distance"]["p5"]]
    x = np.arange(2)
    ax.bar(x - 0.18, med, 0.36, label="median", color="#3b6ea5")
    ax.bar(x + 0.18, p5, 0.36, label="5th percentile", color="#9ec4e8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("L2 distance to nearest training image")
    ax.set_title("Pixel neighbours: 2,048 training references")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGS / "memorisation.png")
    plt.close()


def fig_per_class_accuracy():
    d = load("classifier.json")
    if not d:
        return
    pc = d["per_class_test_acc"]
    order = sorted(pc, key=pc.get)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(order, [pc[k] for k in order], color="#b5651d")
    ax.set_xlim(0, 1)
    ax.set_xlabel("test accuracy")
    ax.set_title("Classifier accuracy per organ")
    plt.tight_layout()
    plt.savefig(FIGS / "per_class_accuracy.png")
    plt.close()


if __name__ == "__main__":
    for fn in [fig_class_distribution, fig_training_curves, fig_samples,
               fig_counterfactual_panel, fig_validity_proximity,
               fig_memorisation, fig_per_class_accuracy]:
        fn()
        print(f"built {fn.__name__}")
