# Changelog

All notable changes to the PRISM dataset, code, and accompanying paper.

## [1.13] — 2026-05

This is a substantial correction-and-extension release. **Several headline
numbers changed relative to the very first public version; if you cited the
earlier release, please update.** The changes below reflect cross-checking
every figure against the source data and adding controlled, reproducible
experiments.

### Corrected (results)

- **PRISM does not beat EMBER on binary detection.** The earliest release
  claimed a large improvement over the EMBER reference. That comparison was not
  sample-controlled. Under a sample-matched, 20-seed controlled benchmark, EMBER
  holds a small but consistent advantage at TPR@FPR=0.1% (mean Δ = +0.85 pp,
  EMBER ahead in 20/20 splits, Wilcoxon p < 1e-4), while the two are
  operationally indistinguishable at the decision threshold (McNemar median
  p = 0.06). PRISM's contribution is **representational efficiency** (5.6× fewer
  dimensions, interpretable per-section structure) and a **quantified
  separability analysis**, not a detection gain. The binary task is saturated.

- **Benign source.** The corpus uses **SOREL-20M** benigns (29,467), not a
  System32/SysWOW64 set. A source-vs-label probe reaches AUC ≈ 0.9999, so the
  *absolute* detection numbers are not field-performance estimates (the
  controlled comparisons are unaffected; see the paper's Limitations).

- **Corpus sizes.** Family-filtered corpus = 49,204 (19,737 malware /
  29,467 benign), 684 families; full deduplicated corpus = 83,633.

- **BODMAS numbers.** Processed PRISM matrices = 57,133 (not 57,293); 538
  families after family filtering; funnel 57,133 → 16,234 (dedup) → 16,231.

- **B1-proxy** is the position-discarded **mean-pooled** 25-dim vector
  (not "global row only").

- **B3** is **PRISM 425-dim** under a single-class temporal probe (BODMAS
  malware temporal split; SOREL benigns random), not an EMBER baseline.

- **ΔI** is computed over **inter-section** cell pairs:
  C(17,2)×25² = 85,000 pairs (not 90,100). Absolute counts unchanged
  (12,854 pairs with ΔI > 0.01); percentages updated (15.1%).

- **B4 single-seed value** at seed 42 is TPR@0.1% = 0.99283. (An earlier
  0.9928-vs-0.9914 discrepancy was traced to a legacy RNG: `np.random.seed`
  (Mersenne Twister) vs `np.random.default_rng` (PCG64); the released model is
  the PCG64/seed-42 one matching the paper.)

### Added

- **Camino A:** BODMAS is now **included** in the controlled cross-representation
  benchmark. Disarmed BODMAS binaries are restored (`benchmark/bodmas_restore.py`)
  and EMBER-extracted (`benchmark/ember_extract_bodmas.py`) at 99.96% success.
  The earlier "BODMAS has no binaries, so it is excluded" rationale is removed.
- **20-seed deterministic robustness study** with paired McNemar tests
  (`benchmark/b4_multiseed.py`).
- **Per-position KS-entropy test** with Holm–Bonferroni correction
  (`analysis/ks_entropy_profile.py`).
- **Feature schema clarified:** physical matrix 17×26; **25 effective features**;
  column 22 reserved/zero; quartiles are Q2–Q4 (Q1 removed, corr > 0.95).

### Notes

- No PE binaries are distributed (see `SECURITY.md`). Feature matrices: CC BY 4.0;
  code: MIT.
