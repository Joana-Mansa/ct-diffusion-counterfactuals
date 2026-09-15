"""Explain one CT slice by counterfactual, end to end.

Give it a test slice and a target organ. It returns the classifier's original
call, the counterfactual that changes that call, and the four numbers that say
whether the counterfactual is worth believing.

The default noising depth is the operating point found by the sweep in
`counterfactual.py`, which is the largest depth where identity correlation with
the original slice is still high and validity has not yet started to fall.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.schedulers import DDIMScheduler

from counterfactual import generate, load_models, metrics
from data import ORGANS, OrganSlices

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "docs" / "figures"

OPERATING_POINT = 200


def pick_operating_point(default=OPERATING_POINT):
    """Highest-validity start_t from the sweep, if the sweep has been run."""
    p = RESULTS / "counterfactual_sweep.json"
    if not p.exists():
        return default
    sweep = json.loads(p.read_text())["sweep"]
    return max(sweep, key=lambda r: r["validity"])["start_t"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="test slice to explain")
    ap.add_argument("--target", default=None,
                    help=f"target organ, one of {ORGANS}. Defaults to the "
                         "classifier's second-most-likely class.")
    ap.add_argument("--start-t", type=int, default=None)
    ap.add_argument("--scale", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", default="inference_example.png")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    FIGS.mkdir(parents=True, exist_ok=True)
    start_t = args.start_t or pick_operating_point()

    unet, clf, n_train_ts = load_models(dev)
    sched = DDIMScheduler(num_train_timesteps=n_train_ts)
    sched.set_timesteps(args.steps)

    ds = OrganSlices("test")
    x0 = ds[args.index][0].unsqueeze(0).to(dev)
    truth = ds[args.index][1]

    with torch.no_grad():
        probs = F.softmax(clf(x0), dim=1)[0].cpu().numpy()
    source = int(probs.argmax())

    if args.target is not None:
        target_idx = ORGANS.index(args.target)
    else:
        # Second most likely class: the nearest decision the model could flip to.
        target_idx = int(np.argsort(probs)[-2])
    target = torch.tensor([target_idx], device=dev)

    xcf = generate(unet, clf, x0, target, sched, start_t, args.scale, dev)
    m = metrics(x0, xcf, clf, torch.tensor([source], device=dev), target)

    with torch.no_grad():
        cf_probs = F.softmax(clf(xcf), dim=1)[0].cpu().numpy()

    report = {
        "index": args.index,
        "ground_truth": ORGANS[truth],
        "original_prediction": ORGANS[source],
        "original_confidence": float(probs[source]),
        "target_organ": ORGANS[target_idx],
        "counterfactual_prediction": ORGANS[int(cf_probs.argmax())],
        "counterfactual_confidence_in_target": float(cf_probs[target_idx]),
        "flipped_to_target": bool(int(cf_probs.argmax()) == target_idx),
        "start_t": start_t,
        "guidance_scale": args.scale,
        "quality": {k: v for k, v in m.items()},
    }
    (RESULTS / "inference_example.json").write_text(json.dumps(report, indent=2))

    print(f"slice {args.index}: truth {ORGANS[truth]}, "
          f"predicted {ORGANS[source]} at p={probs[source]:.3f}")
    print(f"  counterfactual target {ORGANS[target_idx]} at start_t={start_t}")
    print(f"  now predicts {ORGANS[int(cf_probs.argmax())]} "
          f"(target p={cf_probs[target_idx]:.3f}), flipped={report['flipped_to_target']}")
    print(f"  L1 {m['l1']:.4f}  pixels changed {m['frac_pixels_changed']:.3f}  "
          f"identity {m['identity_corr']:.3f}")

    a = x0[0, 0].cpu().numpy()
    b = xcf[0, 0].cpu().numpy()
    diff = b - a
    lim = max(abs(diff).max(), 1e-6)

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1))
    axes[0].imshow(a, cmap="gray", vmin=-1, vmax=1)
    axes[0].set_title(f"original\npredicted {ORGANS[source]} ({probs[source]:.2f})",
                      fontsize=9)
    axes[1].imshow(b, cmap="gray", vmin=-1, vmax=1)
    axes[1].set_title(f"counterfactual\ntoward {ORGANS[target_idx]} "
                      f"({cf_probs[target_idx]:.2f})", fontsize=9)
    axes[2].imshow(diff, cmap="bwr", vmin=-lim, vmax=lim)
    axes[2].set_title(f"difference\nL1 {m['l1']:.3f}, identity "
                      f"{m['identity_corr']:.2f}", fontsize=9)

    order = np.argsort(probs)[::-1][:5]
    y = np.arange(len(order))
    axes[3].barh(y - 0.2, probs[order], 0.4, label="original", color="#3b6ea5")
    axes[3].barh(y + 0.2, cf_probs[order], 0.4, label="counterfactual",
                 color="#b5651d")
    axes[3].set_yticks(y)
    axes[3].set_yticklabels([ORGANS[i] for i in order], fontsize=7)
    axes[3].invert_yaxis()
    axes[3].set_xlim(0, 1)
    axes[3].set_xlabel("probability")
    axes[3].legend(fontsize=7)
    axes[3].set_title("classifier response", fontsize=9)

    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(FIGS / args.out, dpi=140)
    print(f"  wrote {FIGS / args.out}")


if __name__ == "__main__":
    main()
