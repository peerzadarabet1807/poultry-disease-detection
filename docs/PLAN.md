# Plan

Written at the start of the build (Phase 0), after reading the reference repository, the executed notebook
and every training artifact. It records what I found, what I am building and what I need from you.

## What I found

**Training run.** The notebook's outputs name `run_20261029_171740`, and that folder matches it exactly
(TF 2.21.0, Keras 3.13.2, T4, 14.8 min). Every number in the brief checks out against
`results/metrics.json`:

| | n | Accuracy | Macro-F1 | Weighted-F1 | Log loss | ROC-AUC |
|---|---|---|---|---|---|---|
| Internal test (version 2) | 974 | 0.9733 | 0.9599 | 0.9731 | 0.0779 | 0.9984 |
| External PCR (version 3) | 1,201 | 0.7652 | 0.7647 | 0.7651 | 0.9129 | 0.9427 |

External details also match: Coccidiosis recall 0.647, Healthy precision 0.644, NCD to Healthy 44/186
(23.7%), Healthy to Salmonella 70/312 (22.4%). Audit: 8,067 found, 372 rejected (334 exact duplicates,
38 label-conflict files in 19 groups), 7,695 kept, NCD weight 4.339.

**Three more runs in the Drive export.** The zip holds four runs (`run_20260927_131854`,
`run_20260927_151041`, `run_20261029_141903`, `run_20261029_171740`). They have the same configuration,
the same seed and a byte-identical dataset manifest (so the same split); the first two used TF 2.20.0.
They differ only through GPU non-determinism:

| Run | TF | Val macro-F1 | Internal macro-F1 | External acc. | External macro-F1 |
|---|---|---|---|---|---|
| 20260927_131854 | 2.20 | 0.9840 | 0.9690 | 0.7544 | 0.7485 |
| 20260927_151041 | 2.20 | 0.9815 | 0.9597 | 0.7427 | 0.7290 |
| 20261029_141903 | 2.21 | 0.9790 | 0.9572 | 0.7577 | 0.7560 |
| **20261029_171740** (published) | 2.21 | 0.9800 | 0.9599 | **0.7652** | **0.7647** |

The published run has the highest external score of the four. If it was chosen after looking at the
external numbers, the external set was used for model selection and 0.765 is slightly optimistic. I will
publish this run (it is the executed notebook) and report all four runs as a post-hoc run-to-run
variability analysis, so the reader sees the spread (external macro-F1 0.729 to 0.765).

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

## Risks

1. **Antialiased resize parity.** TF's antialiased bilinear is a triangle-filter scale-and-translate. A
   TypeScript port must match it on 3000+ px photos; I will verify it against Python golden tensors.
2. **EXIF.** Training ignored EXIF orientation. Phone photos with an orientation tag will be fed to the
   model unrotated, exactly as in training, which looks odd to users; the UI will show the image as the
   model sees it and explain why.
3. **TF.js size.** ResNet50 in float16 is about 47 MB. uint8 might pass parity; it will be measured.
4. **Docker** needs Docker Desktop running for the image build.
5. **Thresholds** for the "uncertain" state can only be a usability heuristic.

## What I need from you

1. **The sample zip** (STOP after Phase 2 selection): the dataset is on your Drive, not here.
2. **Release confirmation** (STOP before `v1.0.0`).
3. **Was `run_20261029_171740` chosen after seeing external results?** It changes one sentence in the
   README, so I can finish without an answer and fix that sentence later.
4. **Contact email.** The saffron README and `CITATION.cff` include your personal email, but this brief
   forbids personal email addresses in the repo. I am leaving it out and linking your GitHub profile;
   commits use your GitHub noreply address.
5. **gh account.** The active `gh` account is `wiser-ai-admin`. I will switch to `peerzadarabet1807`
   before creating the repo.
