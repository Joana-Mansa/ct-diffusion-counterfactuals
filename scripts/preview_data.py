"""Render the bundled, attributed benchmark examples without downloading data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path(__file__).resolve().parents[1]
d = np.load(R / "examples/data_samples.npz")
meta = json.loads((R / "examples/data_manifest.json").read_text())
if d["images"].ndim == 3:
    fig, axes = plt.subplots(3, 4, figsize=(9, 9))
    for ax, x, item in zip(axes.flat, d["images"], meta["examples"]):
        ax.imshow(x, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{item['label']}\ntest index {item['index']}", fontsize=10)
        ax.axis("off")
    axes.flat[-1].axis("off")
    fig.suptitle("Real OrganAMNIST test samples, 64 x 64 pixels", fontsize=14)
    credit = "First test sample per class. MedMNIST v2 / LiTS, CC BY 4.0."
else:
    fig, axes = plt.subplots(2, 3, figsize=(9, 8))
    for row, (x, item) in enumerate(zip(d["images"], meta["examples"])):
        for axis in range(3):
            ax = axes[row, axis]
            ax.imshow(np.take(x, 32, axis=axis), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"{item['label']} label; test {item['index']}\narray axis {axis}, slice 32", fontsize=10)
            ax.axis("off")
    fig.suptitle("Real NoduleMNIST3D test patches, 64 x 64 x 64 voxels", fontsize=13)
    credit = "Rating-derived labels, not biopsy confirmation. MedMNIST v2 / LIDC-IDRI, CC BY 4.0."
fig.subplots_adjust(top=.88, bottom=.07, hspace=.38, wspace=.12)
fig.text(.5, .025, credit, ha="center", fontsize=9)
fig.savefig(R / "docs/figures/data_samples.png", dpi=160)
