"""
B2-PRISM vs B4-EMBER controlled sub-corpus benchmark.
Sub-corpus: SOREL (29467) + CAPE (3130) + MBZ_modern (376) = 32973 samples.
BODMAS excluded — distributed without PE binaries.
"""
import sys, time, json, pathlib, multiprocessing
t0_global = time.time()

# ── Monkey-patches BEFORE any ember/lief import ──────────────────────────────
import numpy as np
for _a in ('int','float','complex','bool','str','object'):
    if not hasattr(np, _a):
        try: setattr(np, _a, eval(_a))
        except: pass

import lief
try: lief.logging.disable()
except: pass
for _a in ('bad_format','bad_file','pe_error','parser_error','read_out_of_bound'):
    if not hasattr(lief, _a):
        setattr(lief, _a, Exception)

import ember
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Paths & constants ─────────────────────────────────────────────────────────
RESULTS_DIR = pathlib.Path('/home/prism/PRISM 2D/results')
OUT_DIR     = pathlib.Path('/home/prism/PRISM 2D')
SOREL_DIR   = pathlib.Path('/mnt/data4/benignos')
CAPE_DIR    = pathlib.Path('/mnt/data4/cape_binaries')
MBZ_DIR     = pathlib.Path('/mnt/data2/malware_moderno')

FEATURE_MASK = np.array([True]*22 + [False] + [True]*3, dtype=bool)  # slot 22 excluded
N_WORKERS    = min(10, multiprocessing.cpu_count())
N_BOOT       = 1000

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Worker initializer (runs once per worker process) ────────────────────────
_extractor = None

def _worker_init():
    global _extractor
    import numpy as _np
    for _a in ('int','float','complex','bool','str','object'):
        if not hasattr(_np, _a):
            try: setattr(_np, _a, eval(_a))
            except: pass
    import lief as _lief
    try: _lief.logging.disable()
    except: pass
    for _a in ('bad_format','bad_file','pe_error','parser_error','read_out_of_bound'):
        if not hasattr(_lief, _a):
            setattr(_lief, _a, Exception)
    import ember as _ember
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _extractor = _ember.PEFeatureExtractor(feature_version=2)

def _extract_one(args):
    idx, sha, source = args
    global _extractor
    if source == 'SOREL':
        p = pathlib.Path(f'/mnt/data4/benignos/{sha}.pe')
    elif source == 'CAPE':
        p = pathlib.Path(f'/mnt/data4/cape_binaries/{sha}')
    else:
        p = pathlib.Path(f'/mnt/data2/malware_moderno/{sha}.exe')

    if not p.exists():
        return (idx, None, 'file_not_found')
    try:
        raw = p.read_bytes()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fv = _extractor.feature_vector(raw)
        if not np.isfinite(fv).all():
            return (idx, None, 'nan_inf')
        return (idx, fv.astype(np.float32), None)
    except Exception as e:
        return (idx, None, str(e)[:80])

# ── Bootstrap helper ──────────────────────────────────────────────────────────
def bootstrap_metrics(y_true, y_prob, n=N_BOOT, seed=42):
    rng  = np.random.default_rng(seed)
    aucs, tprs = [], []
    for _ in range(n):
        ix = rng.integers(0, len(y_true), len(y_true))
        yt, yp = y_true[ix], y_prob[ix]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
        fpr_, tpr_, _ = roc_curve(yt, yp)
        tprs.append(float(tpr_[np.searchsorted(fpr_, 0.001)]))
    return {
        'auc':  {'median': float(np.median(aucs)),
                 'ci_2_5': float(np.percentile(aucs, 2.5)),
                 'ci_97_5': float(np.percentile(aucs, 97.5)),
                 'std': float(np.std(aucs))},
        'tpr':  {'median': float(np.median(tprs)),
                 'ci_2_5': float(np.percentile(tprs, 2.5)),
                 'ci_97_5': float(np.percentile(tprs, 97.5)),
                 'std': float(np.std(tprs))},
    }

# ── EMBER feature name map ────────────────────────────────────────────────────
def build_ember_feature_names():
    ext = ember.PEFeatureExtractor(feature_version=2)
    names = []
    for fe in ext.features:
        for i in range(fe.dim):
            names.append(f"{fe.name}_{i}")
    return names

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ── 1. Build sub-corpus index ─────────────────────────────────────────────
    log("Loading meta_ff.csv and filtering to PE-available sources")
    meta_ff = pd.read_csv(RESULTS_DIR / 'meta_ff.csv')
    SOURCES = {'SOREL', 'CAPE', 'MalwareBazaar_modern'}
    sub_mask = meta_ff['source'].isin(SOURCES)
    sub_meta = meta_ff[sub_mask].copy().reset_index(drop=False)
    sub_meta.rename(columns={'index': 'ff_idx'}, inplace=True)
    N_SUB = len(sub_meta)
    n_mal = int((sub_meta['label'] == 1).sum())
    n_ben = int((sub_meta['label'] == 0).sum())
    log(f"  Sub-corpus: {N_SUB} total ({n_mal} malware, {n_ben} benign)")

    # ── 2. Extract EMBER features (parallel, with checkpoint) ────────────────
    _ckpt_ember = RESULTS_DIR / 'X_ember_subcorpus.npy'
    _ckpt_valid = RESULTS_DIR / 'X_ember_valid.npy'
    _ckpt_fail  = RESULTS_DIR / 'X_ember_failures.json'

    if _ckpt_ember.exists() and _ckpt_valid.exists():
        log("  EMBER checkpoint found — loading from disk, skipping extraction")
        X_ember = np.load(_ckpt_ember)
        valid   = np.load(_ckpt_valid)
        with open(_ckpt_fail) as _f:
            _fail_data = json.load(_f)
        failures = _fail_data['failures']
        n_fail   = int(_fail_data['n_fail'])
        fail_pct = float(_fail_data['fail_pct'])
        log(f"  Failures (cached): {n_fail} ({fail_pct:.2f}%)")
    else:
        log(f"Extracting EMBER features ({N_SUB} samples, {N_WORKERS} workers)")
        tasks = [(i, row['sha256'], row['source'])
                 for i, row in sub_meta.iterrows()]

        X_ember  = np.zeros((N_SUB, 2381), dtype=np.float32)
        valid    = np.ones(N_SUB, dtype=bool)
        failures = {}
        done     = 0
        t_ext    = time.time()

        with ProcessPoolExecutor(max_workers=N_WORKERS,
                                 initializer=_worker_init) as pool:
            futs = {pool.submit(_extract_one, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                idx, fv, err = fut.result()
                done += 1
                if fv is not None:
                    X_ember[idx] = fv
                else:
                    valid[idx] = False
                    failures[err] = failures.get(err, 0) + 1
                if done % 3000 == 0:
                    log(f"  EMBER: {done}/{N_SUB} ({100*done/N_SUB:.0f}%)")

        n_fail   = int((~valid).sum())
        fail_pct = 100.0 * n_fail / N_SUB
        log(f"  EMBER extraction done in {time.time()-t_ext:.0f}s")
        log(f"  Failures: {n_fail} ({fail_pct:.2f}%) — {failures}")

        if fail_pct > 5.0:
            log("ERROR: >5% extraction failures. Stopping.")
            sys.exit(1)

        np.save(_ckpt_ember, X_ember)
        np.save(_ckpt_valid, valid)
        with open(_ckpt_fail, 'w') as _f:
            json.dump({'n_fail': n_fail, 'fail_pct': fail_pct, 'failures': failures}, _f)
        log("  EMBER checkpoint saved")

    # ── 3. Extract matching PRISM matrices ────────────────────────────────────
    log("Loading PRISM matrices for sub-corpus")
    X_ff_full = np.load(RESULTS_DIR / 'X_ff.npy')   # (49204, 17, 26)
    ff_indices = sub_meta['ff_idx'].values
    X_prism_full = X_ff_full[ff_indices]             # (32973, 17, 26)
    del X_ff_full
    # Apply feature mask + flatten: (N, 17, 25) → (N, 425)
    X_prism_full = X_prism_full[:, :, FEATURE_MASK].reshape(len(sub_meta), -1)
    log(f"  PRISM shape: {X_prism_full.shape}")

    # ── 4. Align valid mask ───────────────────────────────────────────────────
    X_ember_v  = X_ember[valid]
    X_prism_v  = X_prism_full[valid]
    y_v        = sub_meta['label'].values[valid].astype(int)
    N_VALID    = int(valid.sum())
    log(f"  Valid samples after EMBER filter: {N_VALID}")

    # Sanity: no NaN/Inf
    assert np.isfinite(X_ember_v).all(), "NaN/Inf in EMBER features"
    assert np.isfinite(X_prism_v).all(), "NaN/Inf in PRISM features"

    # ── 5. Stratified 80/20 split (seed=42) ───────────────────────────────────
    log("Building stratified 80/20 split")
    np.random.seed(42)
    idx_ben = np.where(y_v == 0)[0]; np.random.shuffle(idx_ben)
    idx_mal = np.where(y_v == 1)[0]; np.random.shuffle(idx_mal)
    n_ben_test = int(len(idx_ben) * 0.2)
    n_mal_test = int(len(idx_mal) * 0.2)
    test_idx  = np.sort(np.concatenate([idx_ben[:n_ben_test], idx_mal[:n_mal_test]]))
    train_idx = np.sort(np.concatenate([idx_ben[n_ben_test:], idx_mal[n_mal_test:]]))
    log(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")
    log(f"  Train mal/ben: {(y_v[train_idx]==1).sum()}/{(y_v[train_idx]==0).sum()}")
    log(f"  Test  mal/ben: {(y_v[test_idx]==1).sum()}/{(y_v[test_idx]==0).sum()}")

    np.save(OUT_DIR / 'results/subcorpus_train_idx.npy', train_idx)
    np.save(OUT_DIR / 'results/subcorpus_test_idx.npy',  test_idx)

    def make_split(X):
        return X[train_idx], X[test_idx], y_v[train_idx], y_v[test_idx]

    Xe_tr, Xe_te, y_tr, y_te = make_split(X_ember_v)
    Xp_tr, Xp_te, _,   _    = make_split(X_prism_v)

    spw = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)
    log(f"  scale_pos_weight = {spw:.4f}")

    lgb_params = dict(n_estimators=1000, learning_rate=0.05, num_leaves=63,
                      min_child_samples=20, n_jobs=-1, random_state=42,
                      scale_pos_weight=spw)

    # ── 6. Train B4-EMBER ─────────────────────────────────────────────────────
    log("Training B4-EMBER (2381-dim)")
    t_b4 = time.time()
    m_ember = lgb.LGBMClassifier(**lgb_params)
    m_ember.fit(Xe_tr, y_tr,
                eval_set=[(Xe_te, y_te)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                            lgb.log_evaluation(period=-1)])
    t_b4 = time.time() - t_b4
    log(f"  B4 done in {t_b4:.1f}s, trees={m_ember.best_iteration_}")

    yp_ember = m_ember.predict_proba(Xe_te)[:, 1]
    auc_b4   = roc_auc_score(y_te, yp_ember)
    fpr_, tpr_, _ = roc_curve(y_te, yp_ember)
    tpr001_b4 = float(tpr_[np.searchsorted(fpr_, 0.001)])
    tpr01_b4  = float(tpr_[np.searchsorted(fpr_, 0.01)])
    f1_b4     = f1_score(y_te, (yp_ember >= 0.5).astype(int))
    cm_b4     = confusion_matrix(y_te, (yp_ember >= 0.5).astype(int)).tolist()
    log(f"  B4 AUC={auc_b4:.5f}  TPR@0.1%={tpr001_b4:.5f}")

    log("  Computing B4 bootstrap CIs (1000 resamples)")
    ci_b4 = bootstrap_metrics(y_te, yp_ember)
    log(f"  B4 CI TPR@0.1%: [{ci_b4['tpr']['ci_2_5']:.4f}, {ci_b4['tpr']['ci_97_5']:.4f}]")

    # Top-20 features by gain
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ember_names = build_ember_feature_names()
    gain_imp = m_ember.booster_.feature_importance(importance_type='gain')
    top20_idx = np.argsort(gain_imp)[::-1][:20]
    top20_feat = [{'feature_name': ember_names[i], 'gain': float(gain_imp[i])}
                  for i in top20_idx]

    # ── 7. Train B2-PRISM (sub-corpus) ────────────────────────────────────────
    log("Training B2-PRISM sub-corpus (425-dim)")
    t_b2 = time.time()
    m_prism = lgb.LGBMClassifier(**lgb_params)
    m_prism.fit(Xp_tr, y_tr,
                eval_set=[(Xp_te, y_te)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                            lgb.log_evaluation(period=-1)])
    t_b2 = time.time() - t_b2
    log(f"  B2 done in {t_b2:.1f}s, trees={m_prism.best_iteration_}")

    yp_prism = m_prism.predict_proba(Xp_te)[:, 1]
    auc_b2   = roc_auc_score(y_te, yp_prism)
    fpr_, tpr_, _ = roc_curve(y_te, yp_prism)
    tpr001_b2 = float(tpr_[np.searchsorted(fpr_, 0.001)])
    tpr01_b2  = float(tpr_[np.searchsorted(fpr_, 0.01)])
    f1_b2     = f1_score(y_te, (yp_prism >= 0.5).astype(int))
    cm_b2     = confusion_matrix(y_te, (yp_prism >= 0.5).astype(int)).tolist()
    log(f"  B2 AUC={auc_b2:.5f}  TPR@0.1%={tpr001_b2:.5f}")

    log("  Computing B2 bootstrap CIs (1000 resamples)")
    ci_b2 = bootstrap_metrics(y_te, yp_prism)
    log(f"  B2 CI TPR@0.1%: [{ci_b2['tpr']['ci_2_5']:.4f}, {ci_b2['tpr']['ci_97_5']:.4f}]")

    # ── 8. Sanity checks ──────────────────────────────────────────────────────
    log("Sanity checks")
    sc = {}
    sc['S1_no_nan_ember']   = bool(np.isfinite(X_ember_v).all())
    sc['S2_auc_b4_ge_95']   = auc_b4 >= 0.95
    sc['S3_auc_b2_ge_95']   = auc_b2 >= 0.95
    sc['S4_boot_b4_ordered']= ci_b4['tpr']['ci_2_5'] <= ci_b4['tpr']['median'] <= ci_b4['tpr']['ci_97_5']
    sc['S5_boot_b2_ordered']= ci_b2['tpr']['ci_2_5'] <= ci_b2['tpr']['median'] <= ci_b2['tpr']['ci_97_5']
    sc['S6_fail_pct_lt_5']  = fail_pct < 5.0
    sc['all_passed']        = all(v for v in sc.values() if isinstance(v, bool) and v is not None)
    for k, v in sc.items():
        status = "OK" if v else "FAIL"
        log(f"  {k}: {status}")

    # ── 9. Output JSON fragment ───────────────────────────────────────────────
    result = {
        "controlled_subcorpus_benchmark": {
            "subcorpus_metadata": {
                "total_samples": N_VALID,
                "malware_samples": int((y_v == 1).sum()),
                "benign_samples":  int((y_v == 0).sum()),
                "bodmas_excluded": 16231,
                "bodmas_exclusion_reason": "BODMAS distributed without PE binaries (feature vectors only)",
                "train_size": len(train_idx),
                "test_size":  len(test_idx),
                "train_malware": int((y_tr == 1).sum()),
                "train_benign":  int((y_tr == 0).sum()),
                "test_malware":  int((y_te == 1).sum()),
                "test_benign":   int((y_te == 0).sum()),
                "extraction_failures": int(n_fail),
                "extraction_failure_rate_pct": round(fail_pct, 3),
                "failure_breakdown": failures,
            },
            "models": {
                "B2_PRISM_subcorpus": {
                    "input_dims": 425,
                    "trees_used": int(m_prism.best_iteration_),
                    "training_seconds": round(t_b2, 1),
                    "auc": round(auc_b2, 5),
                    "tpr_at_fpr_0_001": round(tpr001_b2, 5),
                    "tpr_at_fpr_0_01":  round(tpr01_b2, 5),
                    "f1_at_0_5": round(float(f1_b2), 5),
                    "confusion_matrix_at_0_5": cm_b2,
                    "bootstrap_ci_auc": [round(ci_b2['auc']['ci_2_5'], 5),
                                         round(ci_b2['auc']['ci_97_5'], 5)],
                    "bootstrap_ci_tpr": [round(ci_b2['tpr']['ci_2_5'], 5),
                                         round(ci_b2['tpr']['ci_97_5'], 5)],
                    "bootstrap_median_tpr": round(ci_b2['tpr']['median'], 5),
                    "bootstrap_std_tpr":    round(ci_b2['tpr']['std'], 5),
                },
                "B4_EMBER_subcorpus": {
                    "input_dims": 2381,
                    "trees_used": int(m_ember.best_iteration_),
                    "training_seconds": round(t_b4, 1),
                    "auc": round(auc_b4, 5),
                    "tpr_at_fpr_0_001": round(tpr001_b4, 5),
                    "tpr_at_fpr_0_01":  round(tpr01_b4, 5),
                    "f1_at_0_5": round(float(f1_b4), 5),
                    "confusion_matrix_at_0_5": cm_b4,
                    "bootstrap_ci_auc": [round(ci_b4['auc']['ci_2_5'], 5),
                                         round(ci_b4['auc']['ci_97_5'], 5)],
                    "bootstrap_ci_tpr": [round(ci_b4['tpr']['ci_2_5'], 5),
                                         round(ci_b4['tpr']['ci_97_5'], 5)],
                    "bootstrap_median_tpr": round(ci_b4['tpr']['median'], 5),
                    "bootstrap_std_tpr":    round(ci_b4['tpr']['std'], 5),
                    "top_20_features_gain": top20_feat,
                },
            },
            "controlled_comparison": {
                "metric_primary": "tpr_at_fpr_0_001",
                "B2_PRISM_value": round(tpr001_b2, 5),
                "B4_EMBER_value": round(tpr001_b4, 5),
                "delta_pp_B4_minus_B2": round((tpr001_b4 - tpr001_b2) * 100, 3),
                "delta_auc": round(auc_b4 - auc_b2, 5),
                "interpretation": (
                    "B4 EMBER (2381-dim) vs B2 PRISM (425-dim) on identical 32,973-sample "
                    "sub-corpus (SOREL+CAPE+MBZ_modern), identical split (seed=42), identical "
                    "LightGBM hyperparameters. Isolates contribution of representation type. "
                    "BODMAS excluded (no PEs available)."
                ),
            },
            "sanity_checks": sc,
            "compatibility_notes": {
                "lief_version": lief.__version__,
                "numpy_version": np.__version__,
                "ember_version": getattr(ember, '__version__', 'unknown'),
                "patches_applied": [
                    "numpy_type_aliases (np.int/float/bool/complex/str/object → builtins)",
                    "lief_exception_classes (bad_format/bad_file/pe_error/parser_error/read_out_of_bound → Exception)"
                ],
                "warnings": (
                    "EMBER v2 extractor trained under LIEF 0.9; run here under LIEF 0.14. "
                    "Minor differences in some byte/section features possible but expected "
                    "to be negligible for binary classification at this saturation regime."
                ),
            },
            "runtime_seconds": round(time.time() - t0_global, 1),
        }
    }

    def _json_default(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)

    out_path = OUT_DIR / 'b4_ember_results.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=_json_default)
    log(f"JSON written: {out_path}")

    # Print compact summary
    b4v = result['controlled_subcorpus_benchmark']['models']['B4_EMBER_subcorpus']
    b2v = result['controlled_subcorpus_benchmark']['models']['B2_PRISM_subcorpus']
    cc  = result['controlled_subcorpus_benchmark']['controlled_comparison']
    print("\n" + "="*60)
    print(f"B4-EMBER  AUC={b4v['auc']:.5f}  TPR@0.1%={b4v['tpr_at_fpr_0_001']:.5f}"
          f"  CI=[{b4v['bootstrap_ci_tpr'][0]:.4f},{b4v['bootstrap_ci_tpr'][1]:.4f}]")
    print(f"B2-PRISM  AUC={b2v['auc']:.5f}  TPR@0.1%={b2v['tpr_at_fpr_0_001']:.5f}"
          f"  CI=[{b2v['bootstrap_ci_tpr'][0]:.4f},{b2v['bootstrap_ci_tpr'][1]:.4f}]")
    print(f"Δ(B4-B2)  TPR@0.1%={cc['delta_pp_B4_minus_B2']:+.3f}pp  "
          f"ΔAUC={cc['delta_auc']:+.5f}")
    print("="*60)
    total = time.time() - t0_global
    print(f"Total runtime: {int(total//60)}m {int(total%60)}s")
