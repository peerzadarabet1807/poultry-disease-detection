# Plan

Written at the start of the build (Phase 0), after reading the reference repository, the executed notebook
and every training artifact. It records what I found, what I am building and what I need from you.

## What I found

**Update, 2026-10-30: a fifth run is now the published one.** After this plan was first written, the
notebook was changed to write its outputs straight to Drive and run again. The training code is unchanged
and the dataset manifest, split, class weights and audit files are byte-identical, so
`run_20261029_191001` is a fifth run of the same experiment. The committed notebook documents it, so it is
the published run, and `run_20261029_171740` (the run this plan was first written about) moved to
`results/other_runs/`.

**Training run.** The notebook's outputs name `run_20261029_191001` (TF 2.21.0, Keras 3.13.2, T4,
13.8 min). From `results/metrics.json`:

| | n | Accuracy | Macro-F1 | Weighted-F1 | Log loss | ROC-AUC |
|---|---|---|---|---|---|---|
| Internal test (version 2) | 974 | 0.9671 | 0.9545 | 0.9669 | 0.1021 | 0.9981 |
| External PCR (version 3) | 1,201 | 0.7952 | 0.7945 | 0.7932 | 0.7508 | 0.9524 |

External per class: Healthy precision 0.722, Coccidiosis recall 0.773. Main confusions: Healthy to
Salmonella 65/312 (20.8%), NCD to Healthy 29/186 (15.6%). Audit: 8,067 found, 372 rejected (334 exact
duplicates, 38 label-conflict files in 19 groups), 7,695 kept, NCD weight 4.339. The brief's figures
(~97% vs ~77%, Healthy precision 0.644) describe `run_20261029_171740`; the README uses the published
run's numbers and the five-run spread.

**Five runs with the same configuration.** Same seed, byte-identical dataset manifest (same split); the
first two used TF 2.20.0. They differ only through GPU non-determinism:

| Run | TF | Val macro-F1 | Internal macro-F1 | External acc. | External macro-F1 |
|---|---|---|---|---|---|
| 20260927_131854 | 2.20 | 0.9840 | 0.9690 | 0.7544 | 0.7485 |
| 20260927_151041 | 2.20 | 0.9815 | 0.9597 | 0.7427 | 0.7290 |
| 20261029_141903 | 2.21 | 0.9790 | 0.9572 | 0.7577 | 0.7560 |
| 20261029_171740 | 2.21 | 0.9800 | 0.9599 | 0.7652 | 0.7647 |
| **20261029_191001** (published) | 2.21 | 0.9763 | 0.9545 | **0.7952** | **0.7945** |

The published run has the highest external score of the five and the lowest internal one. Its external
macro-F1 is 1.5 standard deviations above the five-run mean (0.7585, sd 0.0240), so the README reports
the published run together with the spread and does not present 0.795 as typical.

**Dataset licence.** Both Zenodo records are **CC BY 4.0** (version 2: DOI 10.5281/zenodo.4628934;
version 3 PCR: DOI 10.5281/zenodo.5801834; concept DOI 10.5281/zenodo.4495535). Redistribution is
allowed with attribution, so samples and test fixtures can be committed, credited. Funders: IDRC grant
109187-002 (version 2), OWSD grant 4500406715 (version 3).

**A capture-condition clue in the dataset description.** Version 2 says Salmonella and NCD images were
taken from chickens *inoculated* for those diseases (Salmonella one week after, NCD within three days),
while Healthy and Coccidiosis images come from farms. If inoculated birds were housed differently, the
background may correlate with the label in version 2 but not in the PCR set. This is a hypothesis for the
Grad-CAM section, not a fact.

**Paper.** Machuve D., Nwankwo E., Mduma N., Mbelwa J. (2022). *Poultry diseases diagnostics models using
deep learning.* Frontiers in Artificial Intelligence 5:733345. DOI 10.3389/frai.2022.733345 (Crossref).
A corrigendum exists (DOI 10.3389/frai.2022.1016695).

**Environment.** Git 2.55, gh 2.102, Node 24.20 (local; CI uses Node 22 like saffron), Python 3.11.9,
Docker 29.8 (Desktop stopped). TF 2.21.0 and Keras 3.13.2 install and load the `.keras` file on **native
Windows** (CPU), so WSL2 is not needed. `tensorflowjs` goes in a separate venv (saffron pattern).

## Repository layout

As in the brief: `notebooks/`, `results/` (training artifacts, `figures/`, `gradcam/`, `posthoc/`,
`other_runs/`), `models/` (metadata, card, parity, checksums), `src/poultry_disease/` (installable package),
`tests/`, `backend/` (FastAPI + Docker), `web/` (Vite + React + TS + TF.js), `scripts/`, `docs/`,
`.github/workflows/`. Weights (`.keras`, SavedModel zip) go to the GitHub Release only. The TF.js float16
shards are committed under `web/public/model/` (as in saffron), because GitHub Pages must serve them.

**Change, 2026-10-30 (after v1.0.0).** At the owner's request the project now matches saffron: the model runs
only in the browser. `backend/` (FastAPI), the Docker image, `compose.yaml`, the web app's API mode and their
tests and CI job were removed. The Python package, its parity tests and the release weights are unchanged.

## Risks

1. **Antialiased resize parity.** TF's antialiased bilinear is a triangle-filter scale-and-translate. A
   TypeScript port must match it on 3000+ px photos; I will verify it against Python golden tensors.
2. **EXIF.** Training ignored EXIF orientation. Phone photos with an orientation tag will be fed to the
   model unrotated, exactly as in training, which looks odd to users; the UI will show the image as the
   model sees it and explain why.
3. **TF.js size.** ResNet50 in float16 is about 47 MB. uint8 might pass parity; it will be measured.
4. **Thresholds** for the "uncertain" state can only be a usability heuristic.

## What I needed from you (resolved)

1. **The sample images.** No longer needed from Colab: `scripts/fetch_zenodo_images.py` reads single
   images out of the Zenodo zips with HTTP range requests and checks each against the manifest MD5.
2. **Release confirmation** (STOP before `v1.0.0`): still to come.
3. **Run choice.** Publish the run the committed notebook documents (now `run_20261029_191001`) and report
   the others alongside it.
4. **Contact email.** `peerzada.rabet2@gmail.com` goes in the README contact section and `CITATION.cff`;
   commits use the GitHub noreply address.
5. **gh account.** Switch to `peerzadarabet1807` before creating the repo.
