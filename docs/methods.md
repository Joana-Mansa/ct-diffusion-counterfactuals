[Overview](../README.md) · [Data](data.md) · [Methods](methods.md) · [Results](results.md) · [Run it](reproduce.md) · [Verification](verification.md)

# Method

[Architecture diagrams, training losses and learning curves](architecture.md) explain the model components visually.

## Training

1. **Classifier:** OrganCNN learns the 11 organ labels with cross-entropy over 15 epochs. The checkpoint with best validation accuracy is retained.
2. **Generator:** MONAI `DiffusionModelUNet` learns unconditional noise prediction on training slices, with 1,000 diffusion timesteps, 40 epochs, batch size 512 and learning rate 0.00025.
3. **Guided sampling:** noise a real image to `start_t`, denoise with DDIM, and use classifier gradients on the predicted clean image to guide toward an assigned target class.

No paired source/target scans or disease intervention labels are used. This is an implementation and evaluation of established diffusion ideas, not a claim to invent DDPM or classifier guidance.

## What a counterfactual means here

A generated candidate is target-valid if the classifier predicts the requested class. That is a model-behaviour check. It does not establish a physically possible anatomical change or a clinically meaningful explanation. The implementation does not prove a minimum-distance edit.

The sweep selects 256 test slices without replacement with NumPy seed 0. Targets are deterministic classes different from the benchmark label. Of these inputs, 239 are originally classified correctly and one is already classified as its assigned target. This distinction matters when counting genuine prediction flips.

`--steps 100` defines a 100-point DDIM grid across the training schedule. Only grid timesteps at or below `start_t` execute: 6, 10, 15, 20, 30, 40 and 60 updates for levels 50, 100, 150, 200, 300, 400 and 600. It does **not** mean 100 denoising updates at every level. Guidance scale is 6.

## Metric definitions

| Saved field | Definition | Limitation |
|---|---|---|
| `validity` | Fraction whose generated prediction equals the target | Can include an input already predicted as target |
| `flip_rate_any` (original sweep) | Generated prediction differs from the supplied ground-truth label | Not necessarily a change from the original prediction |
| `actual_target_flip_count` (repeat) | Reaches target and original prediction was not target | Counts actual target prediction changes |
| `l1` | Mean absolute pixel difference in [-1, 1] units | Does not measure clinical similarity |
| `frac_pixels_changed` | Fraction with absolute difference >0.1 | Threshold-dependent |
| `identity_corr` | Cosine similarity of mean-centred image vectors (Pearson correlation) | Does not establish anatomical identity |

See [`src/counterfactual.py`](../src/counterfactual.py) for the exact computations. Single-case inference supplies the original predicted class to `metrics`, while the original sweep supplies the dataset label; interpret `flip_rate_any` accordingly.

## Quality and nearest-neighbour checks

The original quality script compares classifier-feature distributions and pixel L2 nearest-neighbour distances. The reported feature Fréchet distance is not standard ImageNet FID and does not independently validate realism. Original nearest-neighbour references were 2,048 sampled training images. The extended verification uses all 34,561 references and all 1,024 generated images recovered from the saved tensor storage.

[Results and caveats](results.md) · [Source](../src/)
