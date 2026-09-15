[Overview](../README.md) · [Data](data.md) · [Methods](methods.md) · [Results](results.md) · [Run it](reproduce.md) · [Verification](verification.md)

# Run and reproduce

## 1. Inspect the included examples

From the repository root, Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
python scripts/preview_data.py
```

This reads `examples/data_samples.npz` and writes `docs/figures/data_samples.png`. It needs no GPU, weights or full dataset. See the [sample attribution and manifest](../examples/README.md).

## 2. Set up model inference

```bash
pip install -r requirements.txt
python scripts/download_checkpoints.py
```

The download script retrieves the audited classifier and DDPM weights (about 45 MB) from the [verified release](https://github.com/Joana-Mansa/ct-diffusion-counterfactuals/releases/tag/verified-2026-09-15). It checks SHA-256 against [`checkpoints.json`](../checkpoints.json) and refuses to replace a different existing file.

The first model run downloads the full OrganAMNIST64 archive (192 MiB). By default it is stored in `data/medmnist`. To reuse an existing archive:

```bash
export MEDMNIST_ROOT=/absolute/path/to/medmnist
```

## 3. Run one example

```bash
python src/infer.py --index 0 --target liver --start-t 200 --seed 2026
```

Outputs: `results/inference_example.json` and `docs/figures/inference_example.png`. These replace the previous demo outputs. The chosen target is a model probe; generation can fail to reach it.

CUDA is selected if available; otherwise the scripts use CPU. Model checks were run on an NVIDIA A100 80 GB. CPU inference is supported by the code but full CPU runtime and minimum GPU memory were not benchmarked. The 3D attribution pipeline can be memory intensive.

## 4. Repeat the verification

```bash
python scripts/verify_checkpoints.py
```

This writes to `verification_run/`, leaving the published experimental results intact. It re-evaluates all 17,778 classifier test inputs, repeats seven guided-sampling levels on 256 inputs, and performs the initial 64-image nearest-neighbour check against all training images. Compare with [`results/verification.json`](../results/verification.json).

The verification repeat seeds PyTorch with 2026 immediately before the noise-level sweep. Its outputs include `replication_per_sample.npz`, `replication_examples.pt` and `verification.json`. Device or library changes can affect floating-point and random-number results.

### Recheck generated-sample quality

```bash
python scripts/verify_sample_quality.py
python scripts/plot_repeat.py
```

The first command uses the explicitly saved 1,024-image `results/generated_samples.npz`, reproduces the original feature-distance and nearest-neighbour statistics, then checks neighbours against all 34,561 training slices. It writes `verification_run/sample_quality_verification.json`. The second command regenerates `docs/figures/replication.png` from the published original and repeat JSON.

## 5. Train new models (optional)

Training writes experiment records and weights. Use a separate clone or back up published result files first. A new training run is not expected to reproduce unrecorded original RNG state exactly.

```bash
python src/train_classifier.py
python src/train_diffusion.py --epochs 40 --batch-size 512 --lr 0.00025 --timesteps 1000
python src/counterfactual.py
python src/evaluate.py
```

Read each script's `--help` before changing settings. For CT-FM, tagged training outputs require explicit checkpoint selection or renaming in a separate analysis workspace; the analysis scripts use the seed-0 default filenames.

## Verified environment and reproducibility limits

The checkpoint audit used Python 3.11, PyTorch 2.14.0+cu130, MONAI 1.6.0, MedMNIST 3.0.2, NumPy 2.4.6 and SciPy 1.17.1. CT-FM also used `lighter-zoo` 0.1.3 and Zennit 1.0.0. [`verified-environment.json`](../results/verified-environment.json) records the environment actually used. Dependency ranges in `requirements.txt` are installation bounds, not a claim that every supported combination was tested.

Full model training was not rerun during this audit. Original stochastic analysis outputs were not all accompanied by saved RNG states. Checkpoint re-evaluation, arithmetic checks and a newly seeded repeat are different levels of evidence; see [verification](verification.md).

## Regenerate architecture drawings and learning curves

```bash
python scripts/architecture_figures.py
```

This needs only NumPy and Matplotlib. It writes SVG and PNG diagrams plus learning curves under `docs/figures/`. The curves read saved JSON histories; no model training runs. Layer shapes and parameter counts are recorded in [`results/architecture.json`](../results/architecture.json).
