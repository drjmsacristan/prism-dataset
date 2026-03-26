# PRISM: PE Relational Inter-Section Matrix

**A 2D Section-Aware Dataset for Static PE Malware Detection**

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

PRISM encodes each Windows PE binary as a 2D matrix
**M ∈ R^{(N_max+1) × F}**, where each row corresponds to one PE
section and a global summary row provides backward compatibility
with EMBER-style models.

Unlike EMBER, BODMAS, and SOREL-20M — which represent each PE
file as a flat 1D vector — PRISM preserves section ordering and
inter-section relational context, enabling classifiers that
exploit structural patterns across consecutive sections.

## Key Results

| Experiment | AUC-ROC | TPR@FPR=0.1% | TPR@FPR=1% |
|---|---|---|---|
| B1 EMBER 2018 (reference) | 0.99338 | 0.8027 | 0.9327 |
| B1-proxy (global row only) | 0.96481 | 0.1136 | 0.5568 |
| B2 PRISM (6K corpus) | 0.99885 | 0.7210 | 0.9843 |
| B2 PRISM full corpus | **0.99979** | **0.9758** | **0.9979** |

PRISM improves on the EMBER 2018 reference by **+17.31 pp** at
TPR@FPR=0.1%, the operationally critical threshold for large-scale deployment.

## Corpus

| Source | Class | Matrices |
|---|---|---|
| BODMAS 2019–2020 | Malware | 57,133 |
| MalwareBazaar 2024–2025 | Malware | 1,681 |
| VirusShare 00499 | Malware | 23,568 |
| **After deduplication** | **Malware** | **33,489** |
| System32 / SysWOW64 | Benign | 4,669 |
| **Total unique** | **Both** | **38,158** |

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
```python
from prism_extractor import extract_prism_matrix

result = extract_prism_matrix("sample.exe")
if result is not None:
    M, mask, n_sections = result
    print(f"Shape: {M.shape}")  # (17, 26)
    print(f"Sections found: {n_sections}")
```

## Feature Set

Each section row contains F=26 features:

| Group | Features | Dim |
|---|---|---|
| Name encoding | Section name hashed | 8 |
| Sizes | SizeOfRawData, VirtualSize, ratio | 3 |
| Permissions | READ, WRITE, EXEC, DISC, CODE, DATA | 6 |
| Entropy | Shannon entropy H | 1 |
| Quartiles | Q1–Q4 over 256-byte windows | 4 |
| Position | Normalised section index | 1 |
| Anomaly | Unusual name, WX flag, zero raw | 3 |

## Dataset Download

The PRISM matrices (.npy files) and trained baseline models
are available on Zenodo:

> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19235865.svg)](https://doi.org/10.5281/zenodo.19235865)

## Citation

If you use PRISM in your research, please cite:
```bibtex
@article{sacristan2026prism,
  title   = {{PRISM}: {PE} Relational Inter-Section Matrix —
             A 2D Section-Aware Dataset for Static PE Malware Detection},
  author  = {Sacrist{\'a}n, Jos{\'e} M. and
             Gonz{\'a}lez-Tablas, Ana I.},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

## License

- Code: [MIT License](LICENSE)
- Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

