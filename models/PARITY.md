# Parity report

One preprocessing contract runs in two places outside the training notebook: the Python package and the
browser. Section 4 covers the input check's image encoder, which runs on the same preprocessed image. This report shows that each one reproduces the notebook, and where it cannot, by how much.
Every number below is read from a file in [`results/parity/`](../results/parity), written by a committed
script or test. These are engineering checks, not model evaluations.

**Fixtures.** 18 original dataset photos in [`tests/fixtures/`](../tests/fixtures) (CC BY 4.0, credited
there), chosen deterministically by [`scripts/select_samples.py`](../scripts/select_samples.py): confident
correct predictions for every class from both test sets, the most confident errors and the two
lowest-margin predictions. Sizes range from 750 x 1000 to 3328 x 1872 pixels. The reference is the
probabilities the notebook recorded for these images on a Colab T4 GPU.

## 1. Python package vs the notebook

[`tests/test_parity.py`](../tests/test_parity.py) and
[`scripts/parity_python.py`](../scripts/parity_python.py) (CPU, TensorFlow 2.21.0, Windows):

| Check | Result |
|---|---|
| Same top class as Colab | **18 / 18** |
| Max absolute probability difference vs Colab | **2.2e-6** (CPU vs GPU arithmetic) |
| In-graph CAM vs the notebook's gradient Grad-CAM, Pearson r (min) | **0.99999999999997** |
| **All 2,175 evaluation images** (974 internal + 1,201 PCR, [`python_vs_colab_external_internal.json`](../results/parity/python_vs_colab_external_internal.json)): same top class | **2,175 / 2,175** |
| All 2,175 evaluation images: max absolute probability difference | **5.7e-6** (median 1.9e-8) |

The full-set check needs the dataset images, fetched from Zenodo and MD5-checked by
[`scripts/fetch_zenodo_images.py`](../scripts/fetch_zenodo_images.py); the per-image results are in
[`python_vs_colab_external_internal.csv`](../results/parity/python_vs_colab_external_internal.csv).

The CAM head is exact by construction: for a GlobalAveragePooling -> Dense head, Grad-CAM equals
ReLU(sum_k w_k A^k) / 49, a 1 x 1 convolution of the last feature map with the Dense kernel
([`src/poultry_disease/model.py`](../src/poultry_disease/model.py)).

## 2. Browser model vs Keras

The browser runs the same graph converted to TF.js ([`scripts/build_web_model.py`](../scripts/build_web_model.py),
[`scripts/convert_to_tfjs.sh`](../scripts/convert_to_tfjs.sh)) after the same preprocessing ported to
TypeScript ([`web/src/lib/resize.ts`](../web/src/lib/resize.ts), [`preprocess.ts`](../web/src/lib/preprocess.ts)).

### 2a. The JPEG decoder is the one thing the browser cannot match

`tf.io.decode_image` decodes JPEGs with libjpeg's **fast** integer IDCT (TensorFlow's default). Browsers,
Pillow and sharp use the **accurate** integer IDCT. On the fixtures the two decodes differ by up to 15 grey
levels per pixel (about 1 level on average), and no browser API selects the fast IDCT. So the browser
cannot see the exact training pixels; it sees what `tf.io.decode_jpeg(dct_method="INTEGER_ACCURATE")`
sees. Section 3 measures what that changes.

### 2b. Preprocessing: exact

| Check | Result |
|---|---|
| TypeScript resize vs `tf.image.resize(antialias=True)`, 5 synthetic images from 100 x 130 to 1152 x 2048 ([`resize.test.ts`](../web/tests/unit/resize.test.ts)) | **bit-identical** |
| Node decode (sharp) vs TensorFlow's accurate-IDCT decode, 18 fixtures | **18 / 18 identical** (SHA-256 of the pixels) |
| 224 x 224 image after the TypeScript resize vs the Python package | **18 / 18 identical** |

### 2c. Conversion: float16 ships, uint8 is rejected

TF.js in Node on the WASM backend, fed by the TypeScript preprocessing, vs Keras on the same decoded
pixels ([`web/tests/parity.node.test.ts`](../web/tests/parity.node.test.ts)):

| TF.js weights | Size | Same top class | Max abs. prob. difference | Mean abs. prob. difference | Min CAM r | Shipped |
|---|---|---|---|---|---|---|
| **float16** ([`tfjs_float16.json`](../results/parity/tfjs_float16.json)) | 47.2 MB, 12 shards | **18 / 18** | **0.0017** | 8.3e-5 | **0.99999** | yes |
| uint8 ([`tfjs_uint8.json`](../results/parity/tfjs_uint8.json)) | 23.7 MB, 6 shards | 17 / 18 | 0.170 | 4.9e-3 | 0.978 | no: flips f17 (Coccidiosis -> Salmonella) |

No tensors leak across the 18 predictions on either variant.

### 2d. End to end: browser path vs training path

Combining the decoder difference with the float16 conversion, against the training pipeline on the same
machine and against Colab:

| Check (float16) | Result |
|---|---|
| Same top class as the training path (Keras, fast-IDCT decode) | **18 / 18** |
| Same top class as Colab | **18 / 18** |
| Max absolute probability difference | 0.104 (f18, a 0.49 / 0.49 Healthy / NCD image the app labels uncertain) |

## 3. How much does the decoder matter?

[`scripts/parity_python.py --decoder-effect`](../scripts/parity_python.py) predicts every evaluation image twice,
from TensorFlow's default (fast IDCT, the training pipeline) and from the accurate IDCT (what a browser sees),
and scores both against the labels:

| Test set | Images | Top class changes | Max abs. prob. difference (median) | Accuracy, training decode | Accuracy, browser decode | Macro-F1, training decode | Macro-F1, browser decode |
|---|---|---|---|---|---|---|---|
| Internal | 974 | 6 (0.6%) | 0.243 (2.6e-7) | 0.967 | 0.967 | 0.954 | 0.952 |
| PCR-confirmed | 1,201 | 15 (1.2%) | 0.238 (1.0e-3) | 0.795 | 0.799 | 0.794 | 0.800 |

The decoder changes the top class on about 1% of photos. All 21 were close calls under the training decode (top-2
margin 0.24 at most, median 0.11), so the app and the package already show "uncertain" for every one of them. It does
not make the model systematically better or worse. Every number in the README and the app is for the training decode, as
recorded by the notebook. A borderline example: the 1024 px copy of f18 in the web app's samples (Healthy, PCR set) is NCD 0.517 / Healthy 0.457 on
the training decode (the Python package) and Healthy 0.575 / NCD 0.405 on the browser decode; both show "uncertain".

## 4. Input check

The input check's image encoder (TinyCLIP ViT-8M/16, see the README's [Input check](../README.md#input-check))
runs in the browser as a TF.js graph model. Its reference is the original PyTorch model with the Hugging Face
weights. Two steps, each checked:

**Keras copy.** The vision tower was rebuilt in Keras with the original weights (exact GELU) and exported as a
SavedModel for the converter ([`build_gate_model.py`](../scripts/build_gate_model.py)). On the 11 sample photos
the Keras embeddings match PyTorch at minimum cosine 0.99999994, and the SavedModel matches Keras exactly
([`gate_keras_tinyclip-vit-8m-16-yfcc15m.json`](../results/parity/gate_keras_tinyclip-vit-8m-16-yfcc15m.json)).

**TF.js vs PyTorch** ([`web/tests/parity.gate.node.test.ts`](../web/tests/parity.gate.node.test.ts), Node, WASM
backend). Fixtures: the 18 dataset photos above and the 12 original OOD photos in
[`docs/ood/`](../docs/ood/README.md); references from
[`make_gate_references.py`](../scripts/make_gate_references.py). "Conversion" feeds TF.js exactly the
224 x 224 image Python made; "end to end" starts from the original JPEG (sharp decode, TypeScript resize).
Required: conversion cosine at least 0.999 and the same pass/fail decision on every fixture in both paths.

| TF.js weights | Size | Conversion: min cosine | Max score difference | Same decision (conversion / end to end) | Shipped |
|---|---|---|---|---|---|
| **float16** ([`gate_float16.json`](../results/parity/gate_float16.json)) | 16.9 MB, 4 shards | **0.999999** | 1.6e-4 | **30 / 30**, **30 / 30** | yes |
| uint8 ([`gate_uint8.json`](../results/parity/gate_uint8.json)) | 8.6 MB, 2 shards | 0.9918 | 0.015 | 30 / 30, 30 / 30 | no: cosine below 0.999 |

End to end, 27 of 30 fixtures decode to exactly the Python image. Three OOD JPEGs (`wood_shavings`,
`pigeon_droppings`, `oat_porridge`) decode to slightly different pixels in sharp's libjpeg-turbo than in
TensorFlow's, so their end-to-end cosine is lower (minimum 0.9959) while the decision stays the same. Browsers
ship their own JPEG decoders, so this kind of difference is expected in the app too; the closest fixture is
0.018 from the threshold. No tensors leak. Reconverting the SavedModel gives byte-identical weight shards; the
`model.json` differs only in the order of attribute keys.

## Reproduce

```bash
python scripts/fetch_zenodo_images.py --selection tests/fixtures/selection.json   # fixtures, MD5-checked
python scripts/build_fixtures.py
pytest tests/test_parity.py                                                      # needs the weights
python scripts/parity_python.py --fixtures --decoder-effect
python scripts/fetch_zenodo_images.py --set internal --set external --workers 6   # ~2.6 GB
python scripts/parity_python.py --set internal --set external --no-gradcam --decoder-effect
python scripts/build_web_model.py && bash scripts/convert_to_tfjs.sh
python scripts/make_web_references.py
PARITY_MODEL_DIR=../build/tfjs/float16 npm --prefix web run test:parity
PARITY_MODEL_DIR=../build/tfjs/uint8 npm --prefix web run test:parity
# input check (see the README for the venvs)
python scripts/build_gate_model.py export tinyclip-vit-8m-16-yfcc15m && python scripts/build_gate_model.py keras tinyclip-vit-8m-16-yfcc15m
bash scripts/convert_to_tfjs.sh --gate tinyclip-vit-8m-16-yfcc15m
python scripts/make_gate_references.py seen && python scripts/make_gate_references.py embed
PARITY_GATE_DIR=../build/gate/tinyclip-vit-8m-16-yfcc15m/tfjs/float16 npm --prefix web run test:parity
PARITY_GATE_DIR=../build/gate/tinyclip-vit-8m-16-yfcc15m/tfjs/uint8 npm --prefix web run test:parity
```
