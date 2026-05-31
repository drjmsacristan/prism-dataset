#!/usr/bin/env python3
"""
ks_entropy_profile.py
=====================
Per-position Kolmogorov-Smirnov test on the PE section entropy distributions
of the PRISM family-filtered corpus (n = 49,204; 19,737 malware + 29,467 benign).

For each section index i in {0, 1, ..., 15}:
    - Collect the entropy values of section i across all benign samples
      where the section is present (mask = 1)
    - Collect the entropy values of section i across all malware samples
      where the section is present
    - Run scipy.stats.ks_2samp (two-sided)
    - Report D_i and the raw p-value

After all 16 tests, apply Holm-Bonferroni correction at alpha = 0.05 to
control the family-wise error rate over the 16 simultaneous tests.

Output: a CSV ks_entropy_results.csv with columns
    position, n_benign, n_malware, D, p_raw, p_holm, reject_holm_05
plus a stdout summary table ready to paste into the paper.

Adjust the paths and column accessors at the top to match your tensor format.
The expected input is a directory of .npz files, each with:
    tensor          shape (17, 25)  -- PRISM matrix
    mask            shape (17,)     -- 1 if row is a real section, 0 if padding
    label           int             -- 0 benign, 1 malware
And the entropy column is at index 17 (entropy slot in the per-section vector;
re-check against your prism-extract schema if this number has shifted).
"""

import numpy as np
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# CONFIG -- ADJUST TO YOUR LAB PATHS
# ---------------------------------------------------------------------------
TENSORS_DIR  = Path("/path/to/tensors_v2")     # family-filtered corpus
ENTROPY_COL  = 17                              # entropy slot index in PRISM row
N_SECTIONS   = 16                              # SEC0..SEC15
ALPHA        = 0.05                            # FWER target

# ---------------------------------------------------------------------------
# 1) GATHER entropy[position] arrays per class
# ---------------------------------------------------------------------------
benign_by_pos  = [[] for _ in range(N_SECTIONS)]
malware_by_pos = [[] for _ in range(N_SECTIONS)]

for path in sorted(TENSORS_DIR.glob("*.npz")):
    z = np.load(path)
    M, mask, y = z["tensor"], z["mask"], int(z["label"])
    bucket = malware_by_pos if y == 1 else benign_by_pos
    for i in range(N_SECTIONS):
        if mask[i] == 1:
            bucket[i].append(float(M[i, ENTROPY_COL]))

# ---------------------------------------------------------------------------
# 2) Per-position two-sample KS test
# ---------------------------------------------------------------------------
results = []
for i in range(N_SECTIONS):
    b = np.asarray(benign_by_pos[i])
    m = np.asarray(malware_by_pos[i])
    if len(b) < 10 or len(m) < 10:
        results.append((i, len(b), len(m), np.nan, np.nan))
        continue
    D, p = stats.ks_2samp(b, m, alternative="two-sided", mode="auto")
    results.append((i, len(b), len(m), float(D), float(p)))

# ---------------------------------------------------------------------------
# 3) Holm-Bonferroni correction over the 16 tests
# ---------------------------------------------------------------------------
valid_idx = [k for k, r in enumerate(results) if not np.isnan(r[4])]
p_raw = [results[k][4] for k in valid_idx]
order = sorted(range(len(p_raw)), key=lambda j: p_raw[j])
m_tests = len(p_raw)
p_holm  = [1.0] * len(p_raw)
running_max = 0.0
for rank, j in enumerate(order):
    adj = (m_tests - rank) * p_raw[j]
    adj = min(adj, 1.0)
    running_max = max(running_max, adj)
    p_holm[j] = running_max

# ---------------------------------------------------------------------------
# 4) Print summary table for the paper
# ---------------------------------------------------------------------------
print(f"\n{'pos':>4} {'n_ben':>8} {'n_mal':>8} {'D':>8} {'p_raw':>12} {'p_holm':>12} {'reject':>8}")
print("-" * 64)
out_rows = []
for k, (i, nb, nm, D, p) in enumerate(results):
    if np.isnan(D):
        print(f"{i:>4} {nb:>8} {nm:>8}   (insufficient n)")
        out_rows.append((i, nb, nm, "", "", "", ""))
        continue
    j = valid_idx.index(k)
    ph = p_holm[j]
    reject = "YES" if ph < ALPHA else "no"
    print(f"{i:>4} {nb:>8} {nm:>8} {D:>8.4f} {p:>12.3e} {ph:>12.3e} {reject:>8}")
    out_rows.append((i, nb, nm, f"{D:.4f}", f"{p:.3e}", f"{ph:.3e}", reject))

# ---------------------------------------------------------------------------
# 5) CSV
# ---------------------------------------------------------------------------
import csv
with open("ks_entropy_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["position", "n_benign", "n_malware", "D", "p_raw", "p_holm",
                "reject_holm_05"])
    w.writerows(out_rows)
print(f"\nSaved ks_entropy_results.csv")
print(f"FWER target alpha = {ALPHA} over {m_tests} simultaneous tests "
      f"(Holm-Bonferroni)")

# ---------------------------------------------------------------------------
# 6) Suggested phrasing for the paper (auto-generated)
# ---------------------------------------------------------------------------
rejected = [r[0] for r in results
            if not np.isnan(r[3]) and p_holm[valid_idx.index(results.index(r))] < ALPHA]
not_rej  = [r[0] for r in results
            if not np.isnan(r[3]) and p_holm[valid_idx.index(results.index(r))] >= ALPHA]

print("\n=== PHRASING SUGGESTION ===")
if rejected:
    print(f"Reject H0 (distributions differ) at Holm-corrected p<{ALPHA} for "
          f"sections {rejected}")
if not_rej:
    print(f"Fail to reject H0 for sections {not_rej}")
print("Insert these numbers into the placeholder in Section V-D.")
