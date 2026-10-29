# Parity report

One preprocessing contract runs in three places: the Python package (and the API built on it), and the
browser. This report shows that each one reproduces the training notebook, and where it cannot, by how much.
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

The CAM head is exact by construction: for a GlobalAveragePooling -> Dense head, Grad-CAM equals
ReLU(sum_k w_k A^k) / 49, a 1 x 1 convolution of the last feature map with the Dense kernel
([`src/poultry_disease/model.py`](../src/poultry_disease/model.py)).

## 2. API vs the notebook

| Path | Same top class | Max abs. probability difference |
|---|---|---|
| FastAPI test client ([`backend/tests/test_api_parity.py`](../backend/tests/test_api_parity.py)) | 18 / 18 | 2.2e-6 |
| Docker image, Linux, `tensorflow-cpu` ([`scripts/parity_api.py`](../scripts/parity_api.py), [`api_vs_colab_docker.json`](../results/parity/api_vs_colab_docker.json)) | **18 / 18** | **1.4e-6** |

## 3. Browser model vs Keras

The browser runs the same graph converted to TF.js ([`scripts/build_web_model.py`](../scripts/build_web_model.py),
[`scripts/convert_to_tfjs.sh`](../scripts/convert_to_tfjs.sh)) after the same preprocessing ported to
TypeScript ([`web/src/lib/resize.ts`](../web/src/lib/resize.ts), [`preprocess.ts`](../web/src/lib/preprocess.ts)).

### 3a. The JPEG decoder is the one thing the browser cannot match

`tf.io.decode_image` decodes JPEGs with libjpeg's **fast** integer IDCT (TensorFlow's default). Browsers,
Pillow and sharp use the **accurate** integer IDCT. On the fixtures the two decodes differ by up to 15 grey
levels per pixel (about 1 level on average), and no browser API selects the fast IDCT. So the browser
cannot see the exact training pixels; it sees what `tf.io.decode_jpeg(dct_method="INTEGER_ACCURATE")`
sees. Section 4 measures what that changes.

### 3b. Preprocessing: exact

| Check | Result |
|---|---|
| TypeScript resize vs `tf.image.resize(antialias=True)`, 5 synthetic images from 100 x 130 to 1152 x 2048 ([`resize.test.ts`](../web/tests/unit/resize.test.ts)) | **bit-identical** |
| Node decode (sharp) vs TensorFlow's accurate-IDCT decode, 18 fixtures | **18 / 18 identical** (SHA-256 of the pixels) |
| 224 x 224 image after the TypeScript resize vs the Python package | **18 / 18 identical** |

### 3c. Conversion: float16 ships, uint8 is rejected

TF.js in Node on the WASM backend, fed by the TypeScript preprocessing, vs Keras on the same decoded
pixels ([`web/tests/parity.node.test.ts`](../web/tests/parity.node.test.ts)):

| TF.js weights | Size | Same top class | Max abs. prob. difference | Mean abs. prob. difference | Min CAM r | Shipped |
|---|---|---|---|---|---|---|
| **float16** ([`tfjs_float16.json`](../results/parity/tfjs_float16.json)) | 47.2 MB, 12 shards | **18 / 18** | **0.0017** | 8.3e-5 | **0.99999** | yes |
| uint8 ([`tfjs_uint8.json`](../results/parity/tfjs_uint8.json)) | 23.7 MB, 6 shards | 17 / 18 | 0.170 | 4.9e-3 | 0.978 | no: flips f17 (Coccidiosis -> Salmonella) |

No tensors leak across the 18 predictions on either variant.

### 3d. End to end: browser path vs training path

Combining the decoder difference with the float16 conversion, against the training pipeline on the same
machine and against Colab:

| Check (float16) | Result |
|---|---|
| Same top class as the training path (Keras, fast-IDCT decode) | **18 / 18** |
| Same top class as Colab | **18 / 18** |
| Max absolute probability difference | 0.104 (f18, a 0.49 / 0.49 Healthy / NCD image the app labels uncertain) |

## 4. How much does the decoder matter?

FULL_SET_DECODER_EFFECT

## Reproduce

```bash
python scripts/fetch_zenodo_images.py --selection tests/fixtures/selection.json   # fixtures, MD5-checked
python scripts/build_fixtures.py
pytest tests/test_parity.py backend/tests/test_api_parity.py                     # needs the weights
python scripts/parity_python.py --fixtures --decoder-effect
python scripts/build_web_model.py && bash scripts/convert_to_tfjs.sh
python scripts/make_web_references.py
PARITY_MODEL_DIR=../build/tfjs/float16 npm --prefix web run test:parity
PARITY_MODEL_DIR=../build/tfjs/uint8 npm --prefix web run test:parity
docker compose up -d && python scripts/parity_api.py --label docker
```
