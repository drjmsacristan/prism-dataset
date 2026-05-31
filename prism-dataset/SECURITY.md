# Security and responsible-use notice

This repository and its Zenodo archive distribute **derived feature data only**
(PRISM matrices, EMBER feature vectors, labels, metadata, trained models). **No
PE executables — benign or malicious — are included.**

Two scripts in `benchmark/` operate on malware binaries that you must obtain
separately:

- `bodmas_restore.py` re-enables a disarmed BODMAS binary by restoring its PE
  `Machine` and `Subsystem` header fields. The output is **functional malware**.
- `ember_extract_bodmas.py` reads those binaries to extract features.

Run these only inside an isolated analysis environment (offline VM, no shared
mounts). Do not redistribute restored binaries. BODMAS binaries are obtained
from the original authors (Yang et al., UIUC) under their data-sharing terms.
