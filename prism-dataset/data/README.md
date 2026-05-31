# Data

The PRISM feature matrices and trained models are **not** stored in this Git
repository (they are large and immutable). Download them from Zenodo:

**DOI: [10.5281/zenodo.19235865](https://doi.org/10.5281/zenodo.19235865)**

Place the downloaded files in this `data/` directory before running the
baselines and analysis scripts.

## Expected contents (from Zenodo)

| File | Shape / rows | Description |
|---|---|---|
| `X_full.npy` | (83,633, 17, 26) float32 | Full deduplicated corpus of PRISM matrices |
| `X_ff.npy` | (49,204, 17, 26) float32 | Family-filtered analysis corpus |
| `y_ff.npy` | (49,204,) int | Labels (0 = benign / SOREL, 1 = malware) |
| `meta_ff.csv` | 49,204 rows | sha256, source, label, family, timestamp_first_seen |
| `X_ember_subcorpus.npy` | (32,973, 2,381) float32 | EMBER vectors for the EMBER-compatible sub-corpus |
| `X_ember_valid.npy` | (32,973,) bool | Validity mask (True = extraction succeeded; 350 False) |
| `X_bodmas_ember.npy` | (57,033, 2,381) float32 | EMBER vectors from **restored** BODMAS binaries |
| `bodmas_ember_idx.npy` | (57,033,) int | Index alignment for the BODMAS EMBER vectors |
| `models/B1_proxy.txt` | — | LightGBM model (mean-pooled 25-dim) |
| `models/B2_prism.txt` | — | LightGBM model (PRISM 425-dim) |
| `models/B3_temporal.txt` | — | LightGBM model (PRISM 425-dim, temporal probe) |
| `models/B4_ember.txt` | — | LightGBM model (EMBER 2,381-dim) |

(Exact filenames may be adjusted on upload; keep this table in sync with the
Zenodo record.)

## Matrix format reminder

Physical shape is **17 × 26**; **column 22 is a reserved, all-zero slot** and
is excluded from analysis, leaving **25 effective features** per row and a
**425-dim** flattened vector. Row 16 is the global summary row.

## No binaries policy

This project distributes **derived feature matrices only**. No PE executables
(benign or malware) are included here or on Zenodo.

- **BODMAS** binaries: request from the original authors (Yang et al., UIUC)
  under their data-sharing terms. They are distributed *disarmed* (PE
  `Machine`/`Subsystem` zeroed); `benchmark/bodmas_restore.py` documents the
  field restoration we applied before EMBER extraction. We do not redistribute
  either the disarmed or the restored binaries.
- **SOREL-20M** (benign source): obtain from Sophos/ReversingLabs.
- **MalwareBazaar** (abuse.ch), **VirusShare**: obtain from their providers.
