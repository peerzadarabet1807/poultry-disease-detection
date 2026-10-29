# Model card: poultry_resnet50 v1.0.0

An image classifier for photos of poultry droppings, with four classes: **Coccidiosis**, **Healthy**,
**Salmonella** and **Newcastle disease (NCD)**. It is a research and decision-support prototype, **not a
veterinary diagnosis**, and it does not replace a veterinarian or a laboratory test.

Every number below is read from a committed file; the source is linked next to it.

## Model details

| | |
|---|---|
| Architecture | ResNet50 (ImageNet weights, `include_top=False`) → GlobalAveragePooling → Dropout(0.3) → Dense(4, softmax) |
| Input | 224 × 224 × 3, ResNet50 "caffe" preprocessing (RGB → BGR, ImageNet means subtracted), applied outside the graph |
| Output | Softmax over `["Coccidiosis", "Healthy", "Salmonella", "NCD"]` ([`class_names.json`](class_names.json)) |
| Training run | `run_20261029_191001`, Colab Tesla T4, TensorFlow 2.21.0, Keras 3.13.2, 13.8 min ([`experiment_config.json`](../results/experiment_config.json)) |
| Weights | GitHub Release `v1.0.0`: `poultry_resnet50.keras` (95 MB) and a SavedModel zip; SHA-256 in [`checksums.sha256`](checksums.sha256) |
| Browser model | Same graph plus a 1 × 1 CAM head, TF.js float16 (47.2 MB) in [`web/public/model/`](../web/public/model) |
| Author | Peerzada Mohammad Rabet |
| License | MIT (code and weights). The dataset is CC BY 4.0 (Machuve et al.). |

**Training.** Two phases with class weights (NCD weight 4.34, [`class_weights.json`](../results/class_weights.json)):
the new head on a frozen backbone (Adam 1e-3, up to 15 epochs), then the `conv5_` stage unfrozen with
BatchNorm kept frozen (AdamW 1e-5, weight decay 1e-4, up to 20 epochs). Augmentation: horizontal and vertical
flips, rotation (±0.08 turns), zoom (±10%), contrast (±15%) and brightness (±10%). Early stopping (patience 5)
and model selection on **validation macro-F1**; the best epoch was 17 (phase 2), validation macro-F1 0.976.

## Intended use

- Research and teaching: how image classifiers behave under domain shift, how to evaluate them honestly,
  and how to explain their output.
- A decision-support prototype that suggests which of four classes a photo resembles, for people who will
  confirm any suspicion with a veterinarian and a laboratory test.

## Out-of-scope uses

- **Veterinary diagnosis**, treatment, vaccination or culling decisions, for a bird or for a flock.
- Deciding whether to report a notifiable disease. Newcastle disease is notifiable in many countries:
  a suspicion goes to a veterinarian or the veterinary authority, whatever the model says.
- Any photo that is not a close-up of poultry droppings (see [Out-of-distribution inputs](#out-of-distribution-inputs)).
- Certifying a flock as healthy. A "Healthy" output is not a clean bill of health.

## Training data

Machuve et al., *Machine Learning Dataset for Poultry Diseases Diagnostics*, Zenodo, CC BY 4.0. Phone photos
taken in the Arusha and Kilimanjaro regions of Tanzania between September 2020 and February 2021.

- **Version 2** ([doi:10.5281/zenodo.4628934](https://doi.org/10.5281/zenodo.4628934)) is used for training,
  validation and the internal test. Healthy and Coccidiosis photos come from poultry farms; Salmonella and
  NCD photos come from chickens **inoculated** with those diseases.
- **Version 3** ([doi:10.5281/zenodo.5801834](https://doi.org/10.5281/zenodo.5801834)) is a separately
  collected set whose labels were confirmed by **PCR** on the matching fecal sample. It is used **only** as an
  external test set, never for training or model selection.

After auditing 8,067 files, 372 were removed: 334 exact duplicates and 38 files in 19 duplicate groups whose
copies carried different labels ([`dataset_audit.json`](../results/dataset_audit.json)). The split is
stratified by class and source, with near-duplicates kept in the same split
([`split_summary.csv`](../results/split_summary.csv)):

| Split | Coccidiosis | Healthy | Salmonella | NCD | Total |
|---|---|---|---|---|---|
| Train (version 2) | 1,378 | 1,440 | 1,467 | 262 | 4,547 |
| Validation (version 2) | 296 | 306 | 315 | 56 | 973 |
| Internal test (version 2) | 296 | 308 | 314 | 56 | 974 |
| External test (version 3, PCR) | 357 | 312 | 346 | 186 | 1,201 |

## Evaluation

Each test set was evaluated once, with the selected model. Intervals are 95% stratified bootstrap intervals of
the saved predictions (10,000 resamples, seed 42, [`bootstrap_ci.json`](../results/posthoc/bootstrap_ci.json));
these and the calibration figures are post-hoc analyses of saved predictions.

| Test set | n | Accuracy | Macro-F1 | Macro ROC-AUC | Log loss |
|---|---|---|---|---|---|
| Internal (version 2) | 974 | **0.967** [0.956, 0.977] | **0.954** [0.935, 0.972] | 0.998 | 0.102 |
| External (version 3, PCR) | 1,201 | **0.795** [0.773, 0.817] | **0.794** [0.770, 0.817] | 0.952 | 0.751 |
| Gap (internal − external) | | 0.172 [0.147, 0.197] | 0.160 [0.130, 0.190] | | |

Per class ([`metrics.json`](../results/metrics.json)):

| Class | Internal precision | Internal recall | External precision | External recall | External n |
|---|---|---|---|---|---|
| Coccidiosis | 0.961 | 1.000 | 0.863 | 0.773 | 357 |
| Healthy | 0.971 | 0.971 | **0.722** | **0.667** | 312 |
| Salmonella | 0.971 | 0.952 | 0.763 | 0.939 | 346 |
| NCD | 0.960 | 0.857 | 0.874 | 0.785 | 186 |

The most common external confusions are Healthy → Salmonella (65 of 312, 20.8%) and NCD → Healthy (29 of 186,
15.6%) ([`external_confusions.json`](../results/posthoc/external_confusions.json)).

**Which number to quote.** The external figure. The internal test set shares photographers, farms, capture
conditions and, for two classes, the inoculation setting with the training data.

**Run-to-run variability.** Five runs with the same configuration, seed and split differ only through GPU
non-determinism. This published run is the best of the five on the external set; across the five, external
accuracy ranges from 0.743 to 0.795 (mean 0.763) and external macro-F1 from 0.729 to 0.794 (mean 0.759, sd
0.024) ([`run_variability.json`](../results/posthoc/run_variability.json)). All five runs agree on 79.4% of
external images and 96.9% of internal ones.

## Calibration

The probabilities are not calibrated for new data ([`calibration.json`](../results/posthoc/calibration.json)):

| | Mean confidence | Accuracy | ECE (95% CI) | Images at ≥ 0.99 confidence | Accuracy at ≥ 0.99 |
|---|---|---|---|---|---|
| Internal | 0.981 | 0.967 | 0.017 [0.012, 0.029] | 89.1% | 0.997 |
| External | 0.915 | 0.795 | 0.120 [0.101, 0.140] | 55.4% | 0.935 |

## Decision thresholds in the app and API

A verdict is shown only when the top probability is at least 0.50 and leads the runner-up by at least 0.90;
otherwise the result is "uncertain". The thresholds were chosen on the internal test set to withhold as many
errors as possible while withholding at most 5% of correct predictions
([`thresholds.json`](../results/posthoc/thresholds.json)). They are a **usability heuristic, not a validated
operating point**. On the external set they show a verdict for 69.5% of images, 90.7% of the shown verdicts are
correct, and 168 of 246 errors are withheld.

## Out-of-distribution inputs

The model has no "not poultry droppings" class. On 12 openly licensed photos of other things, it would show a
confident verdict for 8, including all 5 photos of other animals' droppings
([`docs/ood/`](../docs/ood/README.md)). It must only be used on photos of poultry droppings.

## Known failure modes

1. **Domain shift.** About 17 points of accuracy are lost between the internal and the PCR-confirmed set.
2. **Healthy is the weakest class on new data**: when the model says Healthy on the PCR set, it is right 72.2%
   of the time, and a third of PCR Healthy photos are called something else, mostly Salmonella.
3. **Capture confound (hypothesis).** In version 2, Salmonella and NCD photos come from inoculated birds and the
   other two classes from farms, so the model may have learnt surroundings as well as droppings. Class
   activation maps on some confident PCR-set errors spread over the floor or litter
   ([`cam_internal_vs_pcr.png`](../results/posthoc/cam_internal_vs_pcr.png)); this is consistent with the
   hypothesis on a handful of images, not a measurement.
4. **Overconfidence** on new data (table above).
5. **JPEG decoding.** Training decoded JPEGs with TensorFlow's fast IDCT, which browsers cannot reproduce; on
   borderline photos the browser and the API can rank the top two classes differently. Both then show
   "uncertain" ([`PARITY.md`](PARITY.md)).

## Ethical considerations

- A false "Healthy" can delay care for sick birds and the spread of an infectious disease; a false disease
  pattern can cause needless worry, cost or culling. The interface says "pattern consistent with", never
  "diagnosis", quotes the external Healthy precision, and shows an uncertain state.
- Newcastle disease has legal reporting duties in many countries. Whenever NCD is the most likely class, the
  app and API flag it and advise contacting a veterinarian or veterinary authority promptly.
- The data come from one region of Tanzania. Performance elsewhere, on other breeds, housing or phones, is
  unknown.
- In browser mode photos never leave the device; the API does not store uploads.

## How to cite

See [`CITATION.cff`](../CITATION.cff). Please also cite the dataset (Machuve et al., Zenodo, DOIs above) and the
related paper: Machuve D., Nwankwo E., Mduma N., Mbelwa J. (2022). *Poultry diseases diagnostics models using
deep learning.* Frontiers in Artificial Intelligence 5:733345.
[doi:10.3389/frai.2022.733345](https://doi.org/10.3389/frai.2022.733345)
