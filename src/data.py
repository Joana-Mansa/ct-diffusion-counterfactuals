"""OrganAMNIST loaders.

OrganAMNIST holds axial abdominal CT slices cropped to 11 organs, taken from the
Liver Tumor Segmentation Benchmark. We use the 64x64 release. Images arrive as
uint8 and are scaled to [-1, 1], which is the range the diffusion model expects.
"""

import os
from pathlib import Path

import numpy as np
import torch
from medmnist import OrganAMNIST
from torch.utils.data import DataLoader, Dataset

ORGANS = [
    "bladder", "femur-left", "femur-right", "heart", "kidney-left",
    "kidney-right", "liver", "lung-left", "lung-right", "pancreas", "spleen",
]
SIZE = 64
ROOT = os.environ.get("MEDMNIST_ROOT", str(Path(__file__).resolve().parents[1] / "data" / "medmnist"))


class OrganSlices(Dataset):
    """OrganAMNIST split as float tensors in [-1, 1] with integer organ labels."""

    def __init__(self, split, root=ROOT, size=SIZE):
        Path(root).mkdir(parents=True, exist_ok=True)
        ds = OrganAMNIST(split=split, download=True, root=root, size=size)
        self.imgs = ds.imgs.astype(np.float32) / 127.5 - 1.0
        self.labels = ds.labels.astype(np.int64).squeeze(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.from_numpy(self.imgs[i]).unsqueeze(0), int(self.labels[i])


def loader(split, batch_size, shuffle=None, workers=8, root=ROOT, size=SIZE):
    """DataLoader for one split. Shuffles the train split unless told otherwise."""
    ds = OrganSlices(split, root=root, size=size)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=True, drop_last=shuffle, persistent_workers=workers > 0,
    )


def class_counts(split, root=ROOT, size=SIZE):
    """Per-organ sample counts, used for the dataset table in the README."""
    ds = OrganSlices(split, root=root, size=size)
    counts = np.bincount(ds.labels, minlength=len(ORGANS))
    return {ORGANS[i]: int(c) for i, c in enumerate(counts)}
