[Overview](../README.md) · [Data](data.md) · [Methods](methods.md) · [Results](results.md) · [Run it](reproduce.md) · [Verification](verification.md)

# Data and examples

OrganAMNIST contains organ-centred axial CT crops derived from LiTS. It is an organ-recognition benchmark, not a disease dataset.

## Verified archive

Official [MedMNIST dataset metadata](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py) identifies the [64-pixel archive](https://zenodo.org/records/10519652/files/organamnist_64.npz?download=1). Its MD5 matched the published value during the audit.

| Split | Image array shape | Storage |
|---|---|---|
| train | 34561 × 64 × 64 | uint8 |
| val | 6491 × 64 × 64 | uint8 |
| test | 17778 × 64 × 64 | uint8 |

Images are converted using `image.astype(float32) / 127.5 - 1.0`, then a channel dimension is added. These arrays are resized, quantised benchmark images, not raw Hounsfield-unit volumes or DICOM series.

| Organ | Training slices |
|---|---:|
| bladder | 1956 |
| femur-left | 1390 |
| femur-right | 1357 |
| heart | 1474 |
| kidney-left | 3963 |
| kidney-right | 3817 |
| liver | 6164 |
| lung-left | 3919 |
| lung-right | 3929 |
| pancreas | 3031 |
| spleen | 3561 |

## Inspect a real example

[Sample archive and attribution](../examples/README.md) · [Exact indices, counts and hashes](../examples/data_manifest.json)

The sample archive contains unmodified uint8 images, labels and their original test indices. Examples are chosen by first occurrence of each class, not by model performance. Run `python scripts/preview_data.py` from the repository root to recreate the figure.

## What cannot be audited from this archive

The distributed NPZ does not include patient or site identifiers. This project uses the supplied train/validation/test partition, but does not independently establish patient-level separation, scanner generalisation or external validation.

## Attribution and reuse

MedMNIST v2 data authors retain credit; the included examples are redistributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The panel changes presentation only. Dataset licensing is distinct from model and source-code licensing.
