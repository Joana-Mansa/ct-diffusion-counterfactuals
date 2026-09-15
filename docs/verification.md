[Overview](../README.md) · [Data](data.md) · [Methods](methods.md) · [Results](results.md) · [Run it](reproduce.md) · [Verification](verification.md)

# Verification record

Audit date: **15 September 2026**. The original experiment records are retained, and new verification artifacts are stored separately.

| Evidence | Check performed | Scope |
|---|---|---|
| Data | Archive MD5 against MedMNIST metadata; counts, shapes, ranges and exact sample indices | Original benchmark archive |
| Classifier | Strict checkpoint loading and full test inference | All 17,778 slices; overall and per-class accuracy match |
| Counterfactuals | Seed-2026 repeat, same subset and settings | Seven noise levels, 256 inputs each |
| Actual flips | Compare original prediction, target and generated prediction | Published per-case NPZ |
| Generated-sample quality | Recover all 1,024 images from saved tensor storage; recompute feature distances and pixel neighbours | Original 2,048 references and extended 34,561-reference pool |
| Training curves | Check saved JSON against documentation | Full model training not rerun |

## Material corrections

- Replaced stale per-class README scores with checkpoint-verified values.
- Corrected first-epoch diffusion loss from 0.096 to 0.25735.
- Distinguished a 100-point DDIM grid from actual executed updates.
- Distinguished target attainment from actual prediction flips.
- Replaced an established-optimum claim with the observed stochastic trade-off.
- Removed anatomical-validity and non-memorisation claims unsupported by these checks.

## Traceability

[`results/verification.json`](../results/verification.json) records checkpoint hashes, source hashes at evaluation time and recomputed metrics. [`checkpoints.json`](../checkpoints.json) identifies the distributed weights. [`examples/data_manifest.json`](../examples/data_manifest.json) records the archive hashes and example provenance. Source hashes describe the code used during the audit, before subsequent documentation, path-portability and reporting fixes; they are not asserted to hash the final edited source tree.

The portable [`scripts/verify_checkpoints.py`](../scripts/verify_checkpoints.py) makes a new verification record in `verification_run/`. Full training, patient-level leakage auditing, clinical evaluation and external validation remain outside the completed checks.

## Generated-image storage recovery

The first audit checked the visible 64-image tensor and recorded that scope in `results/verification.json`. A storage-size check then found that `torch.save(gen[:64])` had preserved the complete 1,024-image backing storage. Recovery reproduced all original sample-quality statistics exactly. The first 64 recovered images also matched the visible tensor exactly.

The later, more complete evidence is [`results/sample_quality_verification.json`](../results/sample_quality_verification.json), with an explicit [`generated_samples.npz`](../results/generated_samples.npz) and portable [`verify_sample_quality.py`](../scripts/verify_sample_quality.py). This supersedes the initial limited sample-availability assessment; the original verification JSON remains as a record of the first check.
