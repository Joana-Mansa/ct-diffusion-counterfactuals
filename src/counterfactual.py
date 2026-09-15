"""Generate classifier-guided diffusion candidates and measure pixel changes.

No minimum-edit or anatomical-validity guarantee is implied.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.schedulers import DDIMScheduler

from data import ORGANS, OrganSlices
from train_classifier import OrganCNN
from train_diffusion import build_unet

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"


def load_models(dev):
    """Load the trained DDPM and the classifier it has to fool."""
    ckpt = torch.load(WEIGHTS / "ddpm.pt", map_location=dev)
    unet = build_unet().to(dev)
    unet.load_state_dict(ckpt["state_dict"])
    unet.eval()

    clf = OrganCNN().to(dev)
    clf.load_state_dict(torch.load(WEIGHTS / "classifier.pt", map_location=dev)["state_dict"])
    clf.eval()
    return unet, clf, ckpt["timesteps"]


def generate(unet, clf, x0, target, sched, start_t, scale, dev):
    """Counterfactual for a batch of slices, pushed toward `target` classes.

    Returns the counterfactual batch. The classifier gradient is taken on the
    predicted clean image at each step, which keeps the guidance signal in the
    same space the classifier was trained on.
    """
    n = x0.shape[0]
    noise = torch.randn_like(x0)
    t_start = torch.full((n,), start_t, device=dev, dtype=torch.long)
    x = sched.add_noise(x0, noise, t_start)

    timesteps = [t for t in sched.timesteps if t <= start_t]
    for t in timesteps:
        t_batch = torch.full((n,), int(t), device=dev, dtype=torch.long)
        with torch.no_grad():
            eps = unet(x, t_batch)

        alpha_bar = sched.alphas_cumprod.to(dev)[int(t)]
        x_pred = (x - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()

        with torch.enable_grad():
            xp = x_pred.detach().clone().requires_grad_(True)
            logp = F.log_softmax(clf(xp.clamp(-1, 1)), dim=1)
            sel = logp.gather(1, target.view(-1, 1)).sum()
            grad = torch.autograd.grad(sel, xp)[0]

        eps = eps - scale * (1 - alpha_bar).sqrt() * grad
        x, _ = sched.step(eps, int(t), x)

    return x.clamp(-1, 1)


def metrics(x0, xcf, clf, source, target):
    """Validity, proximity and identity for one batch of counterfactuals."""
    with torch.no_grad():
        pred_cf = clf(xcf).argmax(1)
    flipped = (pred_cf == target).float()
    changed_from_source = (pred_cf != source).float()

    diff = (xcf - x0).flatten(1)
    l1 = diff.abs().mean(1)
    l2 = diff.pow(2).mean(1).sqrt()
    # A pixel counts as edited if it moves more than 5% of the full [-1, 1] range.
    frac_changed = (diff.abs() > 0.1).float().mean(1)

    a = x0.flatten(1)
    b = xcf.flatten(1)
    corr = F.cosine_similarity(a - a.mean(1, keepdim=True),
                               b - b.mean(1, keepdim=True), dim=1)
    return {
        "validity": flipped.mean().item(),
        "flip_rate_any": changed_from_source.mean().item(),
        "l1": l1.mean().item(),
        "l2": l2.mean().item(),
        "frac_pixels_changed": frac_changed.mean().item(),
        "identity_corr": corr.mean().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="test slices to explain")
    ap.add_argument("--scale", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=100, help="points in the full DDIM timestep grid")
    ap.add_argument("--start-ts", type=int, nargs="+",
                    default=[50, 100, 150, 200, 300, 400, 600])
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS.mkdir(exist_ok=True)
    unet, clf, n_train_ts = load_models(dev)

    sched = DDIMScheduler(num_train_timesteps=n_train_ts)
    sched.set_timesteps(args.steps)

    ds = OrganSlices("test")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=args.n, replace=False)
    x0 = torch.stack([ds[i][0] for i in idx]).to(dev)
    source = torch.tensor([ds[i][1] for i in idx], device=dev)

    # Target a different organ chosen deterministically, so reruns match.
    target = (source + 1 + torch.arange(len(source), device=dev) % (len(ORGANS) - 1)) % len(ORGANS)
    target = torch.where(target == source, (target + 1) % len(ORGANS), target)

    with torch.no_grad():
        base_acc = (clf(x0).argmax(1) == source).float().mean().item()
    print(f"classifier accuracy on the {args.n} explained slices: {base_acc:.4f}")

    sweep = []
    for start_t in args.start_ts:
        xcf = generate(unet, clf, x0, target, sched, start_t, args.scale, dev)
        m = metrics(x0, xcf, clf, source, target)
        m["start_t"] = start_t
        m["scale"] = args.scale
        sweep.append(m)
        print(f"start_t {start_t:4d}  validity {m['validity']:.3f}"
              f"  L1 {m['l1']:.4f}  pixels changed {m['frac_pixels_changed']:.3f}"
              f"  identity {m['identity_corr']:.3f}", flush=True)
        torch.save({"x0": x0[:16].cpu(), "xcf": xcf[:16].cpu(),
                    "source": source[:16].cpu(), "target": target[:16].cpu(),
                    "start_t": start_t},
                   RESULTS / f"counterfactual_examples_t{start_t}.pt")

    (RESULTS / "counterfactual_sweep.json").write_text(json.dumps(
        {"n_explained": args.n, "scale": args.scale, "ddim_steps": args.steps,
         "classifier_acc_on_subset": base_acc, "sweep": sweep}, indent=2))


if __name__ == "__main__":
    main()
