# Test fixtures

`images/` holds 18 original, unmodified photos from the dataset, used by the parity tests (Python and
TF.js). `fixtures.json` lists them with the class probabilities the training notebook recorded for each
in Colab, which are the reference the tests compare against.

The selection is deterministic (`scripts/select_samples.py`): per class, confident correct predictions from
both test sets, the most confident errors and the two lowest-margin predictions. The files were fetched
with `scripts/fetch_zenodo_images.py`, which checks each one against the MD5 in
`results/dataset_manifest.csv`, and copied here by `scripts/build_fixtures.py`.

## Attribution

The images are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and are unchanged
apart from the file name (prefixed with the fixture id).

- Files whose `relpath` starts with `version-2/`: Machuve D., Nwankwo E., Mduma N., Mbelwa H., Maguo E.,
  Munisi C. *Machine Learning Dataset for Poultry Diseases Diagnostics* (version 2). Zenodo.
  [doi:10.5281/zenodo.4628934](https://doi.org/10.5281/zenodo.4628934)
- Files whose `relpath` starts with `version-3/`: Machuve D., Nwankwo E., Lyimo E., Maguo E., Munisi C.
  *Machine Learning Dataset for Poultry Diseases Diagnostics - PCR annotated* (version 3). Zenodo.
  [doi:10.5281/zenodo.5801834](https://doi.org/10.5281/zenodo.5801834)
