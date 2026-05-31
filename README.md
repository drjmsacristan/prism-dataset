PRISM: PE Relational Inter-Section Matrix
A 2D Section-Aware Dataset for Static PE Malware Detection
![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)
![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-green.svg)
![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19235865.svg)
PRISM encodes each Windows PE binary as a 2D matrix
M ∈ ℝ^{(N_max+1) × F} in which each row corresponds to one PE
section in file order and an additional global summary row provides
an EMBER-compatible file-level descriptor. Unlike EMBER, BODMAS, and
SOREL-20M — which represent each PE file as a flat 1D feature vector —
PRISM preserves section ordering and inter-section relational context
as first-class, addressable dimensions, enabling per-section
interpretability and architectures that can exploit structural
patterns across sections.
This repository contains the `prism-extract` library, the four baseline
configurations, the controlled cross-representation benchmark, and the
full separability/analysis pipeline. The PRISM feature matrices and
trained models are archived on Zenodo (link above).
---
What PRISM is — and is not
PRISM is a representation and dataset contribution. Its claims are:
Representational efficiency. On a sample-matched, controlled
comparison over 20 deterministic train/test splits, a LightGBM on
the flattened 425-dim PRISM matrix recovers nearly all of the
binary-detection performance of a LightGBM on the 2,381-dim EMBER
vector — at 5.6× fewer input dimensions — conceding only a small,
consistent gap confined to the deep false-positive tail.
Quantified structural content. A formal separability analysis
(Fisher Discriminant Ratio, Mutual Information, inter-cell ΔI,
and per-position Kolmogorov–Smirnov tests) shows the per-section
positional structure carries statistically robust discriminative
information that flat 1D representations discard.
PRISM does not claim to beat EMBER on binary detection. The binary
task is saturated on this corpus (a position-free mean-pooled
aggregate of the per-section features alone approaches the same
ceiling), so the structural content is reserved for tasks with metric
headroom (family classification, 2D-aware architectures, adversarial
robustness, concept drift).
> **Important caveat (read before using the absolute numbers).** Benign
> samples in this corpus come exclusively from SOREL-20M and malware
> never does. A probe trained to predict the *source* (SOREL vs.
> non-SOREL) instead of the label reaches AUC ≈ 0.9999 — essentially
> identical to the malware detector. Source and label are therefore
> near-collinear, so the **absolute** detection metrics here are not
> field-performance estimates. The controlled comparisons (PRISM vs.
> EMBER; positional vs. mean-pooled) are unaffected, because the
> compared models share the identical samples/splits and the confound
> cancels. See the paper's Limitations section.
---
Key results (controlled cross-representation benchmark)
Primary comparison: 20 independent deterministic stratified 80/20 splits
(seeds 42–61) on the 32,623-sample EMBER-compatible sub-corpus, identical
LightGBM hyperparameters, LightGBM forced deterministic
(`num_threads=1, force_row_wise=True, deterministic=True`).
Model	Input dim	TPR@FPR=0.1% (mean ± std over 20 seeds)
B2 PRISM (flattened)	425	0.9887 ± 0.0058
B4 EMBER (1D vector)	2,381	0.9971 ± 0.0019
EMBER ahead in 20/20 splits; mean Δ = +0.85 pp (range +0.14 to +2.15 pp).
Paired difference significant in the deep tail: Wilcoxon p < 1e-4 (paired t-test p ≈ 9e-6).
Operationally indistinguishable at threshold 0.5: paired McNemar median p = 0.06.
Confirmation on the full BODMAS-inclusive corpus (48,825 samples, single
seed): B2 PRISM AUC 0.99965 / TPR@0.1% 0.9975; B4 EMBER AUC 0.99999 /
TPR@0.1% 0.9987 (Δ = +0.127 pp, same direction).
Within-PRISM ablation (positional vs. position-discarded, same features,
full family-filtered corpus):
Model	Input dim	AUC-ROC	TPR@FPR=0.1%
B1-proxy (per-section features mean-pooled, position discarded)	25	0.99980	0.9949
B2 PRISM (flattened, position retained)	425	0.99980	0.9924
The two are indistinguishable on the binary task: a position-free
aggregate already saturates the metric.
---
Corpus
All analyses use the family-filtered corpus. The full deduplicated
corpus is released alongside it for users who do not need family labels.
Stage	Count
Raw matrices combined (all sources, before dedup)	178,740
Unique matrices after global (sha256) deduplication	83,633
Family-filtered primary analysis corpus	49,204
— family-labelled malware	19,737
— SOREL-20M benign	29,467
Distinct malware families	684
Malware sources (family-filtered):
Source	Period	Matrices	Families
BODMAS	2019–2020	16,231	538
CAPE (historical)	—	3,130	154
MalwareBazaar	2024–2025	376	68
Total (sources overlap)	—	19,737	684
Benign: 29,467 unique matrices from the SOREL-20M benign
distribution (disarmed; PE section table preserved for LIEF parsing).
> **BODMAS processing note.** BODMAS is distributed as *disarmed*
> binaries (PE `Machine`/`Subsystem` zeroed). We obtained 57,218 disarmed
> binaries and extracted 57,133 PRISM matrices (85 failures); after global
> dedup these collapse to 16,234 (≈71.6% were cross-source duplicates with
> VirusShare/others) and to 16,231 after family filtering. We do **not**
> redistribute BODMAS binaries; obtain them from the original authors
> (see *Data availability*).
---
Matrix format
Each PRISM matrix has physical shape 17 × 26 (`float32`):
Rows `0…N−1`: the N real PE sections in file order.
Row `16` (`N_max`): the global summary row (5 file-level descriptors
populated: normalised section count, log-imports, log-exports,
has-signatures, has-resources; remaining slots zero).
Rows `N…15`: zero-padded; an accompanying boolean `mask` marks active rows.
Effective features per row: F = 25. Column index 22 is a reserved
slot, always zero (kept for backward compatibility with an earlier
26-feature extractor); it is excluded from all analyses, so the flattened
vector used by the baselines is 17 × 25 = 425 dimensions.
Group	Features	Dim	Column range
Name encoding	Section name hashed to 8-bit vector	8	0–7
Sizes	SizeOfRawData, VirtualSize, raw/virt ratio (log)	3	8–10
Permissions	READ, WRITE, EXEC, DISC, CODE, DATA	6	11–16
Entropy	Shannon entropy H of section bytes	1	17
Quartiles	Q2, Q3, Q4 over 256-byte sliding windows	3	18–20
Position	Normalised section index i/N_max	1	21
(reserved)	zero by construction	1	22
Anomaly	Unusual name; WX co-occurrence; zero raw size	3	23–25
> Note: a first quartile (Q1) feature present in an earlier internal
> version was removed (empirical correlation > 0.95 with entropy and Q2),
> reducing the effective feature count from 26 to 25. The `name*`
> dimensions are bits of a hash projection and are interpretable
> **collectively** (naming regularity at a position), not individually.
---
Installation
```bash
git clone https://github.com/drjmsacristan/prism-dataset.git
cd prism-dataset
pip install -r requirements.txt   # LIEF 0.14.1, numpy, lightgbm, scikit-learn, scipy
```
Quick start (extract a PRISM matrix)
```python
from prism_extract import extract_prism_matrix

result = extract_prism_matrix("sample.exe")
if result is not None:
    M, mask, n_sections = result
    print(M.shape)        # (17, 26) physical; 25 effective features (col 22 reserved)
    print(n_sections)     # number of real PE sections
```
Reproducing the paper
Download the matrices and models from Zenodo into `data/` (see
`data/README.md`), then:
```bash
# Separability analysis (FDR / MI / ΔI / KS) -> results/prism_results.json, ks_entropy_results.csv
python analysis/separability_fdr_mi.py
python analysis/delta_i_pairs.py
python analysis/ks_entropy_profile.py

# Baselines on the full family-filtered corpus
python baselines/b1_proxy.py
python baselines/b2_prism.py
python baselines/b3_temporal.py          # PRISM 425-dim, single-class temporal probe

# Controlled cross-representation benchmark (primary: 20 seeds + paired McNemar)
python benchmark/b4_multiseed.py         # -> results/b4_multiseed_results.json

# Figures
python analysis/make_figures.py          # -> figures/*.png
```
To reproduce the exact numbers, keep LightGBM deterministic
(`num_threads=1, force_row_wise=True, deterministic=True`) and the seeds
as shipped; multi-threaded LightGBM yields slightly different tail metrics.
Repository layout
```
prism_extract/   PRISM feature extractor (LIEF 0.14.1)
baselines/       B1-proxy, B2, B3 baseline training
benchmark/       B4 cross-representation, multiseed, BODMAS restore + EMBER extraction
analysis/        FDR/MI, ΔI, KS-entropy, figure generation
results/         JSON/CSV outputs reproduced by the scripts above
data/            (matrices/models live on Zenodo; see data/README.md)
figures/         generated publication figures
```
Data availability
PRISM matrices and trained baseline models: Zenodo, DOI
10.5281/zenodo.19235865.
BODMAS binaries: not redistributed here; request from the original
authors (Yang et al., UIUC) under their data-sharing terms.
SOREL-20M / MalwareBazaar: obtain from the original providers
(Sophos/ReversingLabs; abuse.ch). We redistribute only derived feature
matrices, never executables.
Citation
```bibtex
@article{sacristan2026prism,
  title   = {{PRISM}: {PE} Relational Inter-Section Matrix --
             A 2D Section-Aware Dataset for Static PE Malware Detection},
  author  = {Sacrist{\'a}n, Jos{\'e} M. and Gonz{\'a}lez-Tablas, Ana I.},
  year    = {2026},
  note    = {Under review}
}
```
License
Code: MIT
Dataset / feature matrices: CC BY 4.0
