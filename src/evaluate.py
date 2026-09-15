"""Sample quality and the memorisation check.

Two things are measured here.

Realism. We report a Frechet distance computed on features from the organ
classifier trained in this repository, not from an ImageNet Inception network.
Inception features are trained on natural photographs and the usual complaint
about applying standard FID to medical images is that the feature space does not
describe them well. Using a feature extractor trained on the same CT data keeps
the measure in the right domain. The number is therefore comparable within this
repository and not against published FID values elsewhere, which is stated again
in the README.

Memorisation. Latent diffusion models trained on medical images have been shown
to reproduce training patients (arXiv:2402.01054). For every generated sample we
find its nearest training image and record the distance. On its own that number
means nothing, so we compute the same statistic for held-out real test images.
If generated samples sit much closer to the training set than real unseen images
do, the model is copying rather than generating.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.networks.schedulers import DDIMScheduler
from scipy import linalg

from data import OrganSlices
from train_classifier import OrganCNN
from train_diffusion import build_unet

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"


@torch.no_grad()
def sample(unet, sched, n, dev, batch=64):
    """Draw n unconditional samples with DDIM."""
    out = []
    for i in range(0, n, batch):
        m = min(batch, n - i)
        x = torch.randn(m, 1, 64, 64, device=dev)
        for t in sched.timesteps:
            eps = unet(x, torch.full((m,), int(t), device=dev, dtype=torch.long))
            x, _ = sched.step(eps, int(t), x)
        out.append(x.clamp(-1, 1).cpu())
        print(f"  sampled {i + m}/{n}", flush=True)
    return torch.cat(out)


@torch.no_grad()
def features(clf, x, dev, batch=256):
    """Penultimate features of the organ classifier."""
    out = []
    for i in range(0, len(x), batch):
        chunk = x[i:i + batch].to(dev)
        h = clf.pool(clf.features(chunk)).flatten(1)
        out.append(h.cpu().numpy())
    return np.concatenate(out)


def frechet(a, b):
    """Frechet distance between two sets of feature vectors."""
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a = np.cov(a, rowvar=False)
    cov_b = np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(cov_a.dot(cov_b), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * covmean))


def nearest_train_distance(query, train, dev, batch=64):
    """Smallest L2 pixel distance from each query image to any training image."""
    train_flat = train.flatten(1).to(dev)
    train_sq = (train_flat ** 2).sum(1)
    best = []
    for i in range(0, len(query), batch):
        q = query[i:i + batch].flatten(1).to(dev)
        d2 = (q ** 2).sum(1, keepdim=True) - 2 * q @ train_flat.T + train_sq[None]
        best.append(d2.clamp_min(0).sqrt().min(1).values.cpu())
    return torch.cat(best).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n-real", type=int, default=2048)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS.mkdir(exist_ok=True)

    ckpt = torch.load(WEIGHTS / "ddpm.pt", map_location=dev)
    unet = build_unet().to(dev)
    unet.load_state_dict(ckpt["state_dict"])
    unet.eval()

    clf = OrganCNN().to(dev)
    clf.load_state_dict(torch.load(WEIGHTS / "classifier.pt", map_location=dev)["state_dict"])
    clf.eval()

    sched = DDIMScheduler(num_train_timesteps=ckpt["timesteps"])
    sched.set_timesteps(args.steps)

    print(f"sampling {args.n_samples} images")
    gen = sample(unet, sched, args.n_samples, dev)
    torch.save(gen[:64], RESULTS / "samples.pt")

    train_ds = OrganSlices("train")
    test_ds = OrganSlices("test")
    rng = np.random.default_rng(0)

    train_idx = rng.choice(len(train_ds), size=args.n_real, replace=False)
    test_idx = rng.choice(len(test_ds), size=min(args.n_samples, len(test_ds)), replace=False)
    real_train = torch.stack([train_ds[i][0] for i in train_idx])
    real_test = torch.stack([test_ds[i][0] for i in test_idx])

    fd_gen = frechet(features(clf, gen, dev), features(clf, real_train, dev))
    fd_real = frechet(features(clf, real_test, dev), features(clf, real_train, dev))
    print(f"frechet distance, generated against train: {fd_gen:.3f}")
    print(f"frechet distance, real test against train: {fd_real:.3f}  (floor)")

    d_gen = nearest_train_distance(gen, real_train, dev)
    d_real = nearest_train_distance(real_test, real_train, dev)
    print(f"nearest-train distance, generated: median {np.median(d_gen):.3f}")
    print(f"nearest-train distance, real test: median {np.median(d_real):.3f}")

    (RESULTS / "sample_quality.json").write_text(json.dumps({
        "n_samples": args.n_samples,
        "ddim_steps": args.steps,
        "frechet_classifier_features": {
            "generated_vs_train": fd_gen,
            "real_test_vs_train": fd_real,
            "note": "computed on organ-classifier features, not ImageNet Inception",
        },
        "memorisation": {
            "generated_nn_distance": {
                "median": float(np.median(d_gen)), "mean": float(d_gen.mean()),
                "p5": float(np.percentile(d_gen, 5)), "min": float(d_gen.min()),
            },
            "real_test_nn_distance": {
                "median": float(np.median(d_real)), "mean": float(d_real.mean()),
                "p5": float(np.percentile(d_real, 5)), "min": float(d_real.min()),
            },
            "ratio_median_generated_over_real": float(np.median(d_gen) / np.median(d_real)),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
