# Diffusion Counterfactuals for CT Organ Classification

Training a denoising diffusion model on abdominal CT, then using it to ask a
classifier what it would take to change its mind, and measuring whether the
answer is a usable explanation.

---

## 1. Goal

A counterfactual explanation answers a question a heatmap cannot: not *where*
the model looked, but *what would have to be different* for the model to decide
otherwise. If a classifier calls a slice "liver", the counterfactual is the
smallest edit that makes it call the slice "spleen". If that edit lands on
anatomy a radiologist would also point at, the explanation is informative. If it
scatters noise across the whole image, it is not.

This project builds the pipeline and measures the second part, which is the part
usually skipped.

Three questions:

1. **Can a diffusion model trained on 34,561 CT slices generate counterfactuals
   that actually flip a classifier?**
2. **What does a successful flip cost?** Every counterfactual trades validity
   against how much of the original image survives. The trade-off is a curve,
   not a number, and where it turns is the useful finding.
3. **Is the generative model reproducing its training data?** Diffusion models
   trained on medical images are known to memorise patients. A synthetic CT
   pipeline that has not been checked for this should not be used.

---

## 2. Data

**OrganAMNIST** from MedMNIST v2, the 64x64 release. Axial abdominal CT slices
cropped to 11 organs, derived from the Liver Tumor Segmentation Benchmark.

| Property | Value |
|---|---|
| Source | [MedMNIST v2](https://medmnist.com/), CC BY 4.0 |
| Modality | Axial abdominal CT |
| Resolution | 64 x 64, single channel |
| Train / val / test | 34,561 / 6,491 / 17,778 |
| Classes | 11 organs |
| Scaling | uint8 mapped to [-1, 1] |

### Class distribution

![class distribution](docs/figures/class_distribution.png)

The split is uneven. Liver has 6,164 training slices and femur-left has 1,390.
Per-class classifier accuracy is reported below rather than a single average,
because the average hides the classes that matter.

---

## 3. Method

Three components.

**Diffusion model.** A MONAI `DiffusionModelUNet`, 10.4M parameters, predicting
the noise added at each of 1,000 timesteps. Attention at the coarsest resolution
only. Trained unconditionally for 40 epochs at batch 512.

**Classifier.** A four-stage CNN, the model under explanation. It is deliberately
small. The object of study is the explanation method, and a larger classifier
would slow the counterfactual search without changing what is measured. Dropout
is kept so the same network can be reused for uncertainty work later.

**Counterfactual generation.** For a real slice and a target organ:

1. Noise the slice forward to timestep `start_t`.
2. Denoise back down with DDIM, 100 steps.
3. At each step, take the classifier gradient with respect to the *predicted
   clean image* rather than the noisy latent, and push the noise prediction
   toward the target class.

Taking the gradient on the predicted clean image matters. The classifier was
trained on clean slices, so evaluating it on a partially noised latent asks it
about inputs it has never seen.

`start_t` controls everything. Low values keep the image intact and rarely
change the prediction. High values change the prediction and destroy the image.
Sweeping it is the experiment.

### Evaluation

A plausible-looking counterfactual is not yet an explanation. Four axes:

| Axis | Measure |
|---|---|
| **Validity** | fraction where the classifier now predicts the target class |
| **Proximity** | mean L1 change, and fraction of pixels moved more than 0.1 |
| **Identity** | centred cosine similarity between original and counterfactual |
| **Realism** | Frechet distance in classifier feature space |

Plus a **memorisation check**: nearest-neighbour distance from generated samples
to the training set, compared against the same statistic for real held-out test
images.

---

## 4. Results

### Diffusion samples

![samples](docs/figures/samples.png)

Unconditional samples after 40 epochs. Organ boundaries, vessels, bright
contrast structures and dark air are all present and in plausible arrangements.

### Training

![training curves](docs/figures/training_curves.png)

Noise-prediction MSE falls from 0.096 to 0.034 over 40 epochs, 23 minutes on one
A100. The classifier reaches **95.8% test accuracy**.

### Classifier, per organ

![per class accuracy](docs/figures/per_class_accuracy.png)

| Organ | Test accuracy |
|---|---|
| heart | 0.885 |
| kidney-left | 0.899 |
| femur-left | 0.946 |
| kidney-right | 0.946 |
| bladder | 0.952 |
| femur-right | 0.968 |
| pancreas | 0.976 |
| spleen | 0.982 |
| liver | 0.992 |
| lung-right | 0.999 |
| lung-left | 1.000 |

Lungs are near perfect, which is expected given they are mostly air against
tissue. Heart and left kidney are hardest.

### The main result: validity against proximity

Classifier accuracy on the 256 explained slices was 0.934.

| `start_t` | validity | L1 change | pixels changed | identity |
|---|---|---|---|---|
| 50 | 0.086 | 0.054 | 0.162 | 0.981 |
| 100 | 0.301 | 0.090 | 0.344 | 0.950 |
| 150 | 0.387 | 0.114 | 0.424 | 0.923 |
| **200** | **0.426** | 0.155 | 0.546 | 0.866 |
| 300 | 0.375 | 0.305 | 0.777 | 0.602 |
| 400 | 0.398 | 0.422 | 0.851 | 0.335 |
| 600 | 0.219 | 0.481 | 0.876 | 0.079 |

![validity and proximity](docs/figures/validity_proximity.png)

**Validity peaks at `start_t = 200` and then falls.** This is the finding worth
recording. The intuition that pushing further up the diffusion trajectory buys
more successful flips is wrong. Past 200 the image degrades faster than the
classifier is persuaded: identity correlation collapses from 0.87 to 0.08 while
validity drops from 0.43 to 0.22.

So there is an operating point, and beyond it the method stops explaining the
classifier and starts repainting the scan. A counterfactual at `start_t = 600`
changes 88% of pixels and retains almost nothing of the original. Whatever it
shows, it is not an explanation of this slice.

### Counterfactual examples

![counterfactual panel](docs/figures/counterfactual_panel.png)

At the operating point. Top row original, middle row counterfactual, bottom row
the difference. Anatomy survives and the edits are localised and structured
rather than uniform noise.

### Inference on a single slice

`src/infer.py` runs the whole path on one test slice: predict, generate a
counterfactual toward the classifier's second-most-likely organ, and report
whether the result is worth believing.

![inference example](docs/figures/inference_example.png)

```
slice 7: truth lung-left, predicted lung-left at p=1.000
  counterfactual target lung-right at start_t=200
  now predicts lung-right (target p=0.716), flipped=True
  L1 0.1770  pixels changed 0.580  identity 0.914
```

The edit fills the air-filled left lung field with tissue-like texture, which is
what distinguishes the two classes in this dataset, and the classifier moves
from 1.000 on lung-left to 0.716 on lung-right while identity correlation stays
at 0.914. The starting depth defaults to the operating point found by the sweep,
read from `results/counterfactual_sweep.json` rather than hard-coded.

### Memorisation

![memorisation](docs/figures/memorisation.png)

| | median | 5th percentile | minimum |
|---|---|---|---|
| Generated to nearest training image | 22.68 | | 11.48 |
| Real test to nearest training image | 25.22 | | 10.71 |

Generated samples sit about 10% closer to the training set than real unseen
images do (ratio 0.899), and the closest generated sample is *further* from the
training set than the closest real test image is. There is no evidence of
patient reproduction at this scale. The check is cheap and should be run on any
medical generative model before its outputs are shared.

Frechet distance in classifier feature space: **46.8** for generated against
train, with a floor of **2.9** for real test against train. Samples are still
distinguishable from real data, which 40 epochs at this resolution should be
expected to give.

---

## 5. Reproducing

### Requirements

```
torch  monai  medmnist  numpy  scipy  matplotlib  einops
```

One GPU. The full pipeline is about 35 minutes on an A100.

### Pipeline

```bash
# 1. Train the diffusion model (23 min on an A100)
PYTHONPATH=src python src/train_diffusion.py --epochs 40 --batch-size 512

# 2. Train the classifier it has to fool (4 min)
PYTHONPATH=src python src/train_classifier.py --epochs 15 --batch-size 256

# 3. Sweep start_t and measure validity, proximity and identity
PYTHONPATH=src python src/counterfactual.py --n 256 --scale 6.0 --steps 100

# 4. Sample quality and the memorisation check
PYTHONPATH=src python src/evaluate.py --n-samples 1024 --steps 100

# 5. Figures
PYTHONPATH=src python src/figures.py
```

Data downloads itself on first run, about 400MB.

---

## 6. Repository layout

```
src/train_diffusion.py    DDPM training
src/train_classifier.py   the model under explanation
src/counterfactual.py     guided generation and the start_t sweep
src/evaluate.py           Frechet distance and memorisation
src/figures.py            every figure in this README
src/data.py               OrganAMNIST loaders
results/                  metrics as JSON
docs/figures/             figures, all regenerable
```

---

## 7. Limitations

**Resolution.** 64x64 single slices, not full CT volumes. This was a deliberate
trade to keep the whole pipeline inside a day on one GPU. At full resolution the
diffusion model would need a latent formulation, and the counterfactual search
would be substantially more expensive. The method transfers; the numbers here
should not be quoted as if they came from full-volume data.

**Validity is 0.43 at best.** Fewer than half the counterfactuals reach the
target organ. Some of that is the difficulty of turning a lung into a pancreas
inside a fixed field of view, which no amount of guidance should achieve. A
conditional diffusion model, rather than an unconditional one steered by
classifier gradients, would likely do better.

**Frechet distance is measured in classifier feature space**, not ImageNet
Inception space. Inception features are trained on natural photographs and
describe CT poorly. Using the organ classifier keeps the measure in the right
domain, but it makes the number comparable only within this repository, not
against published FID values.

**Organ identity is a proxy task.** Flipping "liver" to "spleen" is not a
clinical decision. The pipeline is the contribution; applying it to a
diagnostic label such as malignancy is the next step.

**Single seed.** The sweep was run once. The shape of the curve is clear, but
individual validity numbers carry seed noise.

---

## 8. References

- Jeanneret et al., *Diffusion Models for Counterfactual Explanations*,
  [arXiv:2203.15636](https://arxiv.org/abs/2203.15636)
- Sanchez et al., *Counterfactual Explanations for Medical Image Classification
  and Regression using Diffusion Autoencoder*,
  [arXiv:2408.01571](https://arxiv.org/abs/2408.01571)
- Dar et al., *Unconditional Latent Diffusion Models Memorize Patient Imaging
  Data*, [arXiv:2402.01054](https://arxiv.org/abs/2402.01054)
- Yang et al., *MedMNIST v2*, [medmnist.com](https://medmnist.com/)
