#!/usr/bin/env python3
"""
b4_multiseed.py
===============
Repite el B4-ampliado (B2-PRISM vs B4-EMBER) con N semillas independientes,
LightGBM forzado a modo determinista (num_threads=1, force_row_wise=True).

Por cada semilla:
  - Split estratificado 80/20 con esa semilla
  - Entrena B2 (PRISM 442-dim) y B4 (EMBER 2381-dim), mismos hiperparámetros
  - Calcula TPR@FPR=0.1%, AUC, McNemar pareado B2 vs B4 en test compartido

Salida: b4_multiseed_results.json + tabla resumen por stdout.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import json, time, sys
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import chi2                 # McNemar

# ── Config ────────────────────────────────────────────────────────────────────
N_SEEDS     = 20
OUT_DIR     = Path("/home/prism/PRISM 2D/OPUS")
RESULTS_DIR = Path("/home/prism/PRISM 2D/results")

# Feature mask — same as b4_ember_benchmark.py (drop col 22, zero-variance)
FEATURE_MASK = [i for i in range(26) if i != 22]   # 25 features → 17×25=425 per sample

LGM_PARAMS_BASE = dict(
    n_estimators     = 1000,
    learning_rate    = 0.05,
    num_leaves       = 63,
    min_child_samples= 20,
    # Determinism enforced:
    num_threads      = 1,
    force_row_wise   = True,
    deterministic    = True,
)

def log(msg): print(f"  {msg}", flush=True)

def tpr_at_fpr(y_true, y_prob, fpr_target):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.searchsorted(fpr, fpr_target)
    return float(tpr[min(idx, len(tpr)-1)])

def mcnemar_p(preds_a, preds_b, y_true):
    """McNemar two-sided on paired binary predictions."""
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    b = int(( correct_a & ~correct_b).sum())   # A right, B wrong
    c = int((~correct_a &  correct_b).sum())   # A wrong, B right
    if b + c == 0:
        return 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)    # with continuity correction
    return float(1 - chi2.cdf(stat, df=1))

def run_one_seed(seed, X_prism, X_ember_full, ember_valid_mask, y_all, spw):
    rng = np.random.default_rng(seed)

    # Filter to valid EMBER samples
    y = y_all[ember_valid_mask]
    Xp = X_prism[ember_valid_mask]
    Xe = X_ember_full

    # Stratified 80/20 split
    idx_ben = np.where(y == 0)[0]; rng.shuffle(idx_ben)
    idx_mal = np.where(y == 1)[0]; rng.shuffle(idx_mal)
    n_ben_te = int(len(idx_ben) * 0.2)
    n_mal_te = int(len(idx_mal) * 0.2)
    te_idx = np.sort(np.concatenate([idx_ben[:n_ben_te], idx_mal[:n_mal_te]]))
    tr_idx = np.sort(np.concatenate([idx_ben[n_ben_te:], idx_mal[n_mal_te:]]))

    y_tr, y_te = y[tr_idx], y[te_idx]
    Xp_tr, Xp_te = Xp[tr_idx], Xp[te_idx]
    Xe_tr, Xe_te = Xe[tr_idx], Xe[te_idx]

    spw_s = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)

    def train(X_tr, X_te):
        params = {**LGM_PARAMS_BASE, 'random_state': seed, 'scale_pos_weight': spw_s}
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])
        return m.predict_proba(X_te)[:, 1], m.best_iteration_

    yp_b2, trees_b2 = train(Xp_tr, Xp_te)
    yp_b4, trees_b4 = train(Xe_tr, Xe_te)

    tpr_b2 = tpr_at_fpr(y_te, yp_b2, 0.001)
    tpr_b4 = tpr_at_fpr(y_te, yp_b4, 0.001)
    auc_b2 = roc_auc_score(y_te, yp_b2)
    auc_b4 = roc_auc_score(y_te, yp_b4)

    # McNemar: hard predictions at 0.5
    pred_b2 = (yp_b2 >= 0.5).astype(int)
    pred_b4 = (yp_b4 >= 0.5).astype(int)
    mc_p    = mcnemar_p(pred_b2, pred_b4, y_te)

    return {
        "seed":    seed,
        "n_train": int(len(tr_idx)),
        "n_test":  int(len(te_idx)),
        "tpr_b2":  round(tpr_b2, 5),
        "tpr_b4":  round(tpr_b4, 5),
        "delta":   round(tpr_b4 - tpr_b2, 5),
        "auc_b2":  round(auc_b2, 5),
        "auc_b4":  round(auc_b4, 5),
        "trees_b2": trees_b2,
        "trees_b4": trees_b4,
        "mcnemar_p": round(mc_p, 4),
        "ci_overlap": None,   # computed post-hoc if needed
    }


def main():
    print(f"Loading data (same checkpoints as b4_ember_benchmark.py)...", flush=True)

    import pandas as pd
    meta_ff  = pd.read_csv(RESULTS_DIR / 'meta_ff.csv')
    SOURCES  = {'SOREL', 'CAPE', 'MalwareBazaar_modern'}
    sub_mask = meta_ff['source'].isin(SOURCES)
    sub_meta = meta_ff[sub_mask].copy().reset_index(drop=False)
    sub_meta.rename(columns={'index': 'ff_idx'}, inplace=True)

    # Load EMBER checkpoint (32973, 2381) and validity mask
    X_ember_raw = np.load(RESULTS_DIR / 'X_ember_subcorpus.npy').astype(np.float32)
    valid       = np.load(RESULTS_DIR / 'X_ember_valid.npy')     # bool (32973,)

    # Load PRISM and apply feature mask (drop col 22)
    X_ff_full   = np.load(RESULTS_DIR / 'X_ff.npy')              # (49204, 17, 26)
    ff_indices  = sub_meta['ff_idx'].values
    X_prism_raw = X_ff_full[ff_indices][:, :, FEATURE_MASK].reshape(len(sub_meta), -1)
    del X_ff_full

    y_sub = sub_meta['label'].values.astype(int)

    # Apply validity filter
    X_ember = X_ember_raw[valid]
    X_prism = X_prism_raw[valid]
    y       = y_sub[valid]

    spw = float((y == 0).sum()) / max((y == 1).sum(), 1)
    print(f"Samples: {len(y)} (mal={( y==1).sum()}, ben={(y==0).sum()}), spw={spw:.3f}")
    print(f"Running {N_SEEDS} seeds with deterministic LightGBM (num_threads=1)...\n")

    results = []
    for i, seed in enumerate(range(42, 42 + N_SEEDS)):
        t0 = time.time()
        r = run_one_seed(seed, X_prism, X_ember, np.ones(len(y), dtype=bool), y, spw)
        elapsed = time.time() - t0
        results.append(r)
        ci_b2 = "overlaps" if r['tpr_b2'] > 0.995 else "wide"
        print(f"  Seed {seed:3d}: B2={r['tpr_b2']:.4f} B4={r['tpr_b4']:.4f} "
              f"Δ={r['delta']:+.4f} McNemar_p={r['mcnemar_p']:.4f}  [{elapsed:.0f}s]",
              flush=True)

    # Summary
    deltas  = np.array([r['delta']  for r in results])
    tpr_b2s = np.array([r['tpr_b2'] for r in results])
    tpr_b4s = np.array([r['tpr_b4'] for r in results])
    mc_ps   = np.array([r['mcnemar_p'] for r in results])

    summary = {
        "n_seeds":      N_SEEDS,
        "tpr_b2_mean":  round(float(tpr_b2s.mean()), 5),
        "tpr_b2_std":   round(float(tpr_b2s.std()),  5),
        "tpr_b4_mean":  round(float(tpr_b4s.mean()), 5),
        "tpr_b4_std":   round(float(tpr_b4s.std()),  5),
        "delta_mean":   round(float(deltas.mean()),  5),
        "delta_std":    round(float(deltas.std()),   5),
        "delta_min":    round(float(deltas.min()),   5),
        "delta_max":    round(float(deltas.max()),   5),
        "pct_delta_gt0": round(float((deltas > 0).mean()), 3),
        "mcnemar_p_median": round(float(np.median(mc_ps)), 4),
        "mcnemar_p_min":    round(float(mc_ps.min()), 4),
        "mcnemar_p_max":    round(float(mc_ps.max()), 4),
    }

    print(f"\n{'='*60}")
    print(f"  RESUMEN ({N_SEEDS} semillas)")
    print(f"{'='*60}")
    print(f"  B2 TPR@0.1%:  {summary['tpr_b2_mean']:.4f} ± {summary['tpr_b2_std']:.4f}")
    print(f"  B4 TPR@0.1%:  {summary['tpr_b4_mean']:.4f} ± {summary['tpr_b4_std']:.4f}")
    print(f"  Δ TPR:        {summary['delta_mean']:+.4f} ± {summary['delta_std']:.4f}")
    print(f"  Δ rango:      [{summary['delta_min']:+.4f}, {summary['delta_max']:+.4f}]")
    print(f"  Δ > 0:        {summary['pct_delta_gt0']*100:.0f}% de las semillas")
    print(f"  McNemar p:    {summary['mcnemar_p_median']:.4f} mediana "
          f"[{summary['mcnemar_p_min']:.4f}, {summary['mcnemar_p_max']:.4f}]")
    print(f"{'='*60}\n")

    out = {"summary": summary, "per_seed": results}
    outfile = OUT_DIR / "b4_multiseed_results.json"
    outfile.write_text(json.dumps(out, indent=2))
    print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
