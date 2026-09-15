# Real benchmark examples

`data_samples.npz` contains a small, deterministic selection from the official MedMNIST 64-pixel release: the first test example of each class. These are real processed benchmark inputs, not diffusion-generated images.

```python
import numpy as np
samples = np.load("examples/data_samples.npz")
print(samples.files)  # images, labels, test_indices
print(samples["images"].shape, samples["images"].dtype)
image = samples["images"][0]
model_input = image.astype("float32") / 127.5 - 1.0
```

[The manifest](data_manifest.json) records the source URL, official archive MD5, SHA-256, split counts and exact sample indices. `python scripts/preview_data.py` regenerates the README preview using NumPy and Matplotlib only.

## Attribution

Source: Jiancheng Yang and colleagues, **MedMNIST v2**, Scientific Data 10, 41 (2023), [project and citations](https://medmnist.com/). Derived from LiTS (OrganAMNIST).

These sample arrays are redistributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The preview re-renders their grayscale values and adds labels. The model preprocessing rescales uint8 values to [-1, 1]. This notice applies to the data examples and does not assign a licence to third-party models.
