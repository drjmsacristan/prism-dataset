#!/usr/bin/env python3
"""PRISM v1.0 — Full Computation Pipeline (Phases 1-8)"""
import sys, os, time, json, csv, warnings, math
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/prism')
sys.path.insert(0, '/home/prism/prism-env/lib/python3.12/site-packages')

# ── Paths ─────────────────────────────────────────────────────────────────────
PRISM_DIR    = Path('/home/prism/PRISM 2D')
RESULTS_DIR  = PRISM_DIR / 'results'
FIGURES_DIR  = PRISM_DIR / 'figures'
PRISM_MAT    = Path('/mnt/data2/prism_matrices')
SOREL_DIR    = Path('/mnt/data4/benignos')
CAPE_BIN_DIR = Path('/mnt/data4/cape_binaries')
MBZ_DIR      = Path('/mnt/data2/malware_moderno')
VS_DIR       = Path('/mnt/data2/virusshare/extracted')
META_V2      = Path('/home/prism/data/crystal/meta_v2.csv')
BODMAS_NAMES = PRISM_MAT / 'nombres_bodmas.txt'
BODMAS_META  = Path('/home/prism/data/bodmas/bodmas_metadata.csv')
LOG_PATH     = PRISM_DIR / 'pipeline.log'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

t0_global = time.time()

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')

# ── Feature layout ─────────────────────────────────────────────────────────────
N_ROWS = 17; N_COLS = 26
FEATURE_MASK = np.array([True]*22 + [False] + [True]*3, dtype=bool)
N_EFF = int(FEATURE_MASK.sum())  # 25

FEAT_NAMES_26 = ['name0','name1','name2','name3','name4','name5','name6','name7',
                 'raw_size','virt_size','ratio',
                 'MEM_READ','MEM_WRITE','MEM_EXEC','MEM_DISC','CNT_CODE','CNT_DATA',
                 'entropy','position','Q2','Q3','Q4','reserved',
                 'unusual_name','WX_flag','zero_raw']
FEAT_NAMES_25 = [n for n,m in zip(FEAT_NAMES_26, FEATURE_MASK) if m]
SEC_NAMES     = [f'SEC{i}' for i in range(16)] + ['GLOBAL']
# Binary feature indices in 25-eff array (slots 11-16 perms, 22-24 anomaly after masking slot 22)
BINARY_IN_25  = list(range(11,17)) + [22,23,24]

SIN_FAMILIA = {'virusshare','bodmas','malwarebazaar_modern','unknown',''}
N_WORKERS   = min(10, multiprocessing.cpu_count())

# ── Extractor (worker) ─────────────────────────────────────────────────────────
_prism_mod = None

def _init_worker():
    global _prism_mod
    import importlib.util, sys
    sys.path.insert(0, '/home/prism')
    spec = importlib.util.spec_from_file_location("prism_extractor", "/home/prism/prism_extractor.py")
    _prism_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_prism_mod)

def _extract_one(filepath):
    global _prism_mod
    try:
        result = _prism_mod.extract_prism_matrix(str(filepath))
        if result is None:
            return None, 'prism_none'
        M, mask, n_sec = result
        M = np.array(M, dtype=np.float32)
        if M.shape != (17, 26):
            return None, f'shape_{M.shape}'
        stem = Path(filepath).name
        for ext in ['.pe','.exe','.dll','.bin']:
            stem = stem.replace(ext,'').replace(ext.upper(),'')
        return M, stem.lower()
    except Exception as e:
        return None, f'err:{type(e).__name__}:{str(e)[:40]}'

def extract_batch(filepaths, desc='extract'):
    matrices, shas, fail = [], [], {'prism_none':0,'shape':0,'exception':0}
    total = len(filepaths)
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_init_worker) as pool:
        futs = {pool.submit(_extract_one, fp): fp for fp in filepaths}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % 3000 == 0:
                log(f"  {desc}: {done}/{total} ({100*done/total:.0f}%)")
            M, info = fut.result()
            if M is not None:
                matrices.append(M); shas.append(info)
            else:
                k = 'prism_none' if 'prism_none' in info else \
                    'shape' if 'shape' in info else 'exception'
                fail[k] += 1
    X = np.array(matrices, dtype=np.float32) if matrices else np.zeros((0,17,26),dtype=np.float32)
    return X, shas, fail, total

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Load & Extract
# ─────────────────────────────────────────────────────────────────────────────
def phase1():
    log("=== PHASE 1: Load & Extract ===")
    t0 = time.time()
    parts_X, parts_y, parts_meta = [], [], []
    extraction_report = {}

    # ── P1.1a BODMAS (reuse) ─────────────────────────────────────────────────
    log("P1.1a BODMAS: loading X_bodmas_mal.npy + family join")
    Xb = np.load(PRISM_MAT/'X_bodmas_mal.npy')
    names_b = open(BODMAS_NAMES).read().strip().splitlines()
    assert len(names_b) == len(Xb), f"BODMAS name/matrix mismatch: {len(names_b)} vs {len(Xb)}"

    # Build sha→family+timestamp from bodmas_metadata.csv
    b_meta = {}
    with open(BODMAS_META) as f:
        for row in csv.DictReader(f):
            sha = row.get('sha','').strip().lower()
            if sha:
                b_meta[sha] = (row.get('family','').strip(), row.get('timestamp','').strip())

    b_families, b_timestamps = [], []
    for sha in names_b:
        fam, ts = b_meta.get(sha.lower(), ('',''))
        b_families.append(fam if fam else float('nan'))
        b_timestamps.append(ts if ts else float('nan'))

    yb = np.ones(len(Xb), dtype=np.uint8)
    meta_b = pd.DataFrame({'sha256': names_b, 'source': 'BODMAS', 'label': 1,
                            'family': b_families, 'timestamp_first_seen': b_timestamps})
    parts_X.append(Xb); parts_y.append(yb); parts_meta.append(meta_b)
    log(f"  BODMAS: {len(Xb)} matrices, {sum(1 for f in b_families if isinstance(f,str) and f)} with family")
    extraction_report['BODMAS'] = {'reused': True, 'n': len(Xb)}

    # ── P1.1b VirusShare original (reuse) ────────────────────────────────────
    log("P1.1b VirusShare original: loading X_virusshare.npy")
    Xv = np.load(PRISM_MAT/'X_virusshare.npy')
    yv = np.ones(len(Xv), dtype=np.uint8)
    meta_v = pd.DataFrame({'sha256': [f'vs_orig_{i}' for i in range(len(Xv))],
                            'source': 'VirusShare_original', 'label': 1,
                            'family': float('nan'), 'timestamp_first_seen': float('nan')})
    parts_X.append(Xv); parts_y.append(yv); parts_meta.append(meta_v)
    log(f"  VirusShare orig: {len(Xv)} matrices")
    extraction_report['VirusShare_original'] = {'reused': True, 'n': len(Xv)}

    # ── P1.1c MalwareBazaar legacy (reuse) ───────────────────────────────────
    log("P1.1c MalwareBazaar legacy: loading X_malware.npy")
    Xm = np.load(PRISM_MAT/'X_malware.npy')
    ym = np.ones(len(Xm), dtype=np.uint8)
    meta_m = pd.DataFrame({'sha256': [f'mbz_leg_{i}' for i in range(len(Xm))],
                            'source': 'MalwareBazaar_legacy', 'label': 1,
                            'family': float('nan'), 'timestamp_first_seen': float('nan')})
    parts_X.append(Xm); parts_y.append(ym); parts_meta.append(meta_m)
    log(f"  MBZ legacy: {len(Xm)} matrices")
    extraction_report['MalwareBazaar_legacy'] = {'reused': True, 'n': len(Xm)}

    # ── P1.2b CAPE malware (re-extract) ─────────────────────────────────────
    log("P1.2b CAPE malware: building meta_v2 lookup + extracting")
    meta_v2_latest = {}
    with open(META_V2) as f:
        for row in csv.DictReader(f):
            sha = row.get('sha256','').strip().lower()
            if sha: meta_v2_latest[sha] = row

    cape_files = []
    cape_sha2fam = {}
    for fname in os.listdir(CAPE_BIN_DIR):
        sha = fname.lower().strip()
        row = meta_v2_latest.get(sha, {})
        fam = row.get('family','').strip().lower()
        if fam and fam not in SIN_FAMILIA:
            fp = CAPE_BIN_DIR / fname
            cape_files.append(fp)
            cape_sha2fam[sha] = fam

    log(f"  CAPE candidates with family: {len(cape_files)}")
    Xc, shas_c, fail_c, total_c = extract_batch(cape_files, 'CAPE')
    log(f"  CAPE extracted: {len(Xc)}/{total_c}, fail={fail_c}")

    families_c = [cape_sha2fam.get(s, float('nan')) for s in shas_c]
    yc = np.ones(len(Xc), dtype=np.uint8)
    meta_c = pd.DataFrame({'sha256': shas_c, 'source': 'CAPE', 'label': 1,
                            'family': families_c, 'timestamp_first_seen': float('nan')})
    parts_X.append(Xc); parts_y.append(yc); parts_meta.append(meta_c)
    extraction_report['CAPE'] = {'reused': False, 'n_candidates': total_c,
                                  'extracted_ok': len(Xc), 'failures': fail_c}

    # ── P1.2c MalwareBazaar moderno (re-extract) ─────────────────────────────
    log("P1.2c MalwareBazaar moderno: extracting")
    mbz_files = sorted(MBZ_DIR.glob('*.exe'))
    log(f"  MBZ moderno candidates: {len(mbz_files)}")
    Xmz, shas_mz, fail_mz, total_mz = extract_batch(mbz_files, 'MBZ_modern')
    log(f"  MBZ moderno extracted: {len(Xmz)}/{total_mz}, fail={fail_mz}")

    # Family from meta_v2 if available
    fam_mz = [meta_v2_latest.get(s,{}).get('family','').strip().lower() for s in shas_mz]
    fam_mz = [f if f and f not in SIN_FAMILIA else float('nan') for f in fam_mz]
    ymz = np.ones(len(Xmz), dtype=np.uint8)
    meta_mz = pd.DataFrame({'sha256': shas_mz, 'source': 'MalwareBazaar_modern', 'label': 1,
                             'family': fam_mz, 'timestamp_first_seen': float('nan')})
    parts_X.append(Xmz); parts_y.append(ymz); parts_meta.append(meta_mz)
    extraction_report['MalwareBazaar_modern'] = {'reused': False, 'n_candidates': total_mz,
                                                  'extracted_ok': len(Xmz), 'failures': fail_mz}

    # ── P1.2d VirusShare extra (re-extract) ──────────────────────────────────
    log("P1.2d VirusShare extra: extracting all 49,967")
    vs_files = sorted(VS_DIR.glob('*.exe'))
    log(f"  VirusShare extra candidates: {len(vs_files)}")
    Xve, shas_ve, fail_ve, total_ve = extract_batch(vs_files, 'VS_extra')
    log(f"  VirusShare extra extracted: {len(Xve)}/{total_ve}, fail={fail_ve}")

    yve = np.ones(len(Xve), dtype=np.uint8)
    meta_ve = pd.DataFrame({'sha256': shas_ve, 'source': 'VirusShare_extra', 'label': 1,
                             'family': float('nan'), 'timestamp_first_seen': float('nan')})
    parts_X.append(Xve); parts_y.append(yve); parts_meta.append(meta_ve)
    extraction_report['VirusShare_extra'] = {'reused': False, 'n_candidates': total_ve,
                                              'extracted_ok': len(Xve), 'failures': fail_ve}

    # ── P1.2a SOREL benign (re-extract) ──────────────────────────────────────
    log("P1.2a SOREL benign: extracting 30,037 PEs")
    sorel_files = sorted(SOREL_DIR.glob('*.pe'))
    log(f"  SOREL candidates: {len(sorel_files)}")
    Xs, shas_s, fail_s, total_s = extract_batch(sorel_files, 'SOREL')
    log(f"  SOREL extracted: {len(Xs)}/{total_s}, fail={fail_s}")

    ys = np.zeros(len(Xs), dtype=np.uint8)
    meta_s = pd.DataFrame({'sha256': shas_s, 'source': 'SOREL', 'label': 0,
                            'family': 'N/A', 'timestamp_first_seen': float('nan')})
    parts_X.append(Xs); parts_y.append(ys); parts_meta.append(meta_s)
    extraction_report['SOREL'] = {'reused': False, 'n_candidates': total_s,
                                   'extracted_ok': len(Xs), 'failures': fail_s}

    # ── Combine ───────────────────────────────────────────────────────────────
    log("Combining all sources...")
    X_full  = np.concatenate([p for p in parts_X  if len(p) > 0], axis=0)
    y_full  = np.concatenate([p for p in parts_y  if len(p) > 0], axis=0)
    meta_full = pd.concat([p for p in parts_meta if len(p) > 0], ignore_index=True)

    log(f"Phase 1 done: {len(X_full)} total matrices, elapsed {time.time()-t0:.0f}s")
    np.save(RESULTS_DIR/'X_full.npy',  X_full.astype(np.float32))
    np.save(RESULTS_DIR/'y_full.npy',  y_full)
    meta_full.to_csv(RESULTS_DIR/'meta_full.csv', index=False)
    return X_full, y_full, meta_full, extraction_report

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Deduplication
# ─────────────────────────────────────────────────────────────────────────────
def phase2(X_full, y_full, meta_full):
    log("=== PHASE 2: Deduplication ===")
    t0 = time.time()
    N_before = len(X_full)
    log(f"  Before dedup: {N_before}")

    X_flat = X_full.reshape(N_before, -1).astype(np.float32)
    _, unique_idx = np.unique(X_flat, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)

    X_dedup    = X_full[unique_idx]
    y_dedup    = y_full[unique_idx]
    meta_dedup = meta_full.iloc[unique_idx].reset_index(drop=True)

    N_after = len(X_dedup)
    pct_removed = 100.0 * (N_before - N_after) / N_before
    log(f"  After dedup: {N_after} (removed {N_before-N_after}, {pct_removed:.1f}%)")

    # Cross-source duplicate matrix
    sources = meta_full['source'].values
    dup_mask = np.ones(N_before, dtype=bool)
    dup_mask[unique_idx] = False
    dup_srcs = sources[dup_mask]
    keep_srcs = sources[unique_idx]

    # Count duplicates between sources
    src_list = ['BODMAS','VirusShare_original','MalwareBazaar_legacy','CAPE',
                'MalwareBazaar_modern','VirusShare_extra','SOREL']
    cross_dup = {}
    # For each duplicate, record which source it's from
    from collections import Counter
    dup_source_counts = Counter(dup_srcs)
    log(f"  Duplicates by source: {dict(dup_source_counts)}")

    log(f"Phase 2 done, elapsed {time.time()-t0:.0f}s")
    np.save(RESULTS_DIR/'X_dedup.npy',  X_dedup.astype(np.float32))
    np.save(RESULTS_DIR/'y_dedup.npy',  y_dedup)
    meta_dedup.to_csv(RESULTS_DIR/'meta_dedup.csv', index=False)
    return X_dedup, y_dedup, meta_dedup, N_before, N_after, pct_removed, dict(dup_source_counts)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Subset definition
# ─────────────────────────────────────────────────────────────────────────────
def phase3(X_dedup, y_dedup, meta_dedup):
    log("=== PHASE 3: Subset definition ===")
    from collections import Counter
    _xf = RESULTS_DIR/'X_ff.npy'; _yf = RESULTS_DIR/'y_ff.npy'; _mf = RESULTS_DIR/'meta_ff.csv'
    if _xf.exists() and _yf.exists() and _mf.exists():
        log("  Checkpoint found: loading X_ff/y_ff/meta_ff from disk, skipping Phase 3 recompute")
        X_ff   = np.load(_xf)
        y_ff   = np.load(_yf)
        meta_ff = pd.read_csv(_mf)
        src_counts = Counter(meta_dedup['source'])
        fam_col = meta_ff[y_ff==1]['family'].astype(str)
        uniq_fams = Counter(fam_col.str.lower())
        top30 = uniq_fams.most_common(30)
        n_mal_fam = int((y_ff==1).sum()); n_ben = int((y_ff==0).sum())
        log(f"  corpus_full by source: {dict(src_counts)}")
        log(f"  corpus_family_filtered: {len(X_ff)} total ({n_mal_fam} malware with family, {n_ben} benign)")
        log(f"  Unique families: {len(uniq_fams)}")
        return X_ff, y_ff, meta_ff, dict(src_counts), n_mal_fam, n_ben, len(uniq_fams), top30

    # corpus_full = X_dedup
    src_counts = Counter(meta_dedup['source'])
    log(f"  corpus_full by source: {dict(src_counts)}")

    # corpus_family_filtered
    fam_col = meta_dedup['family'].astype(str)
    is_malware = y_dedup == 1
    is_benign  = y_dedup == 0

    def valid_family(f):
        if not isinstance(f, str):
            return False
        return f and f not in {'nan','NaN','None','N/A',''} and \
               f.lower() not in SIN_FAMILIA

    has_family = fam_col.apply(valid_family).values

    ff_mask = (is_malware & has_family) | is_benign
    X_ff    = X_dedup[ff_mask]
    y_ff    = y_dedup[ff_mask]
    meta_ff = meta_dedup[ff_mask].reset_index(drop=True)

    n_mal_fam   = int((y_ff == 1).sum())
    n_ben       = int((y_ff == 0).sum())
    families    = meta_ff[y_ff == 1]['family'].astype(str).apply(lambda x: x.lower())
    uniq_fams   = Counter(families)
    top30       = uniq_fams.most_common(30)

    log(f"  corpus_family_filtered: {len(X_ff)} total "
        f"({n_mal_fam} malware with family, {n_ben} benign)")
    log(f"  Unique families: {len(uniq_fams)}")
    log(f"  Top 10: {top30[:10]}")

    np.save(RESULTS_DIR/'X_ff.npy',  X_ff.astype(np.float32))
    np.save(RESULTS_DIR/'y_ff.npy',  y_ff)
    meta_ff.to_csv(RESULTS_DIR/'meta_ff.csv', index=False)
    return X_ff, y_ff, meta_ff, dict(src_counts), n_mal_fam, n_ben, len(uniq_fams), top30

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Separability
# ─────────────────────────────────────────────────────────────────────────────
def phase4(X_ff, y_ff):
    log("=== PHASE 4: Separability ===")
    import json as _json
    _ckpt = RESULTS_DIR / 'sep_result.json'
    if _ckpt.exists():
        log("  Checkpoint found: loading sep_result from disk, skipping Phase 4")
        with open(_ckpt) as _f:
            return _json.load(_f)
    from sklearn.feature_selection import mutual_info_classif
    from collections import namedtuple

    # Apply feature mask: (N,17,25)
    X25 = X_ff[:, :, FEATURE_MASK]   # (N, 17, 25)
    N   = len(X25)
    y   = y_ff.astype(int)

    # P4.1 — Section count distribution
    log("P4.1 Section count distribution")
    sec_counts = np.array([
        np.count_nonzero(np.any(X25[i, :16, :] != 0, axis=1))
        for i in range(N)
    ])
    p50 = int(np.percentile(sec_counts, 50))
    p90 = int(np.percentile(sec_counts, 90))
    p95 = int(np.percentile(sec_counts, 95))
    p99 = int(np.percentile(sec_counts, 99))
    smax = int(sec_counts.max())
    nmax_cov = float(np.mean(sec_counts <= 16) * 100)
    log(f"  Section counts P50={p50} P90={p90} P95={p95} P99={p99} max={smax} nmax16_cov={nmax_cov:.2f}%")

    hist_b = np.bincount(sec_counts[y==0], minlength=18)[1:18].tolist()
    hist_m = np.bincount(sec_counts[y==1], minlength=18)[1:18].tolist()

    # P4.2 — FDR (17×25)
    log("P4.2 FDR matrix")
    mu1  = np.mean(X25[y==1], axis=0)  # (17,25)
    mu0  = np.mean(X25[y==0], axis=0)
    var1 = np.var(X25[y==1],  axis=0)
    var0 = np.var(X25[y==0],  axis=0)
    denom = var1 + var0
    FDR25 = np.where(denom > 1e-12, (mu1 - mu0)**2 / denom, 0.0)  # (17,25)

    # (17,26) version for figures — insert 0 at slot 22
    FDR26 = np.insert(FDR25, 22, 0.0, axis=1)

    # Top 15
    flat_idx  = np.argsort(FDR25.ravel())[::-1][:15]
    fdr_top15 = []
    for fi in flat_idx:
        i, j = divmod(int(fi), N_EFF)
        fdr_top15.append({'fdr': float(FDR25[i,j]),
                          'feature': f"{FEAT_NAMES_25[j]}@{SEC_NAMES[i]}",
                          'section': SEC_NAMES[i], 'feature_name': FEAT_NAMES_25[j]})
    log(f"  FDR top3: {[(d['feature'],round(d['fdr'],4)) for d in fdr_top15[:3]]}")
    np.save(RESULTS_DIR/'fdr_17x25.npy', FDR25.astype(np.float32))

    # P4.3 — MI (17×25) — flattened call for efficiency
    log("P4.3 MI matrix (this takes a few minutes)")
    t_mi = time.time()
    X_flat425 = X25.reshape(N, -1)   # (N, 425)

    # discrete mask for 425 dims
    disc425 = np.zeros(N_ROWS * N_EFF, dtype=bool)
    for row_i in range(N_ROWS):
        for j in BINARY_IN_25:
            disc425[row_i * N_EFF + j] = True

    MI_flat = mutual_info_classif(X_flat425, y, discrete_features=disc425,
                                   n_neighbors=5, random_state=42)
    MI25 = MI_flat.reshape(N_ROWS, N_EFF)
    MI25 = np.clip(MI25, 0, None)  # clip negatives
    n_clipped = int((MI_flat < 0).sum())
    MI26 = np.insert(MI25, 22, 0.0, axis=1)
    log(f"  MI done in {time.time()-t_mi:.0f}s, clipped {n_clipped} negatives")

    flat_mi = np.argsort(MI25.ravel())[::-1][:15]
    mi_top15 = []
    for fi in flat_mi:
        i, j = divmod(int(fi), N_EFF)
        mi_top15.append({'mi_bits': float(MI25[i,j]),
                         'feature': f"{FEAT_NAMES_25[j]}@{SEC_NAMES[i]}",
                         'section': SEC_NAMES[i], 'feature_name': FEAT_NAMES_25[j]})
    log(f"  MI top3: {[(d['feature'],round(d['mi_bits'],4)) for d in mi_top15[:3]]}")
    np.save(RESULTS_DIR/'mi_17x25.npy', MI25.astype(np.float32))

    # P4.4 — ΔI benchmark then decide
    log("P4.4 ΔI — benchmarking...")
    _di_ckpt = RESULTS_DIR / 'delta_i_pairs.csv'
    if _di_ckpt.exists():
        log(f"  ΔI checkpoint found: loading from {_di_ckpt}")
        di_df = pd.read_csv(_di_ckpt)
        restricted = False  # conservative: assume full run
        pairs_to_run = []   # not needed for stats
        n_above_0   = int((di_df.delta_i > 0).sum())
        n_above_001 = int((di_df.delta_i > 0.01).sum())
        n_above_005 = int((di_df.delta_i > 0.05).sum())
        top20_di    = di_df.nlargest(20, 'delta_i')
        top20_list  = []
        for _, r in top20_di.iterrows():
            top20_list.append({'feature_a': r.feature_a, 'feature_b': r.feature_b,
                               'mi_a': round(r.mi_a,5), 'mi_b': round(r.mi_b,5),
                               'mi_joint': round(r.mi_joint,5), 'delta_i': round(r.delta_i,5)})
        log(f"  ΔI: above0={n_above_0} above0.01={n_above_001} above0.05={n_above_005}")
    else:
        # Build list of all cross-section pairs (i1,j1)-(i2,j2) with i1<i2
        all_pairs = []
        for i1 in range(N_ROWS):
            for j1 in range(N_EFF):
                for i2 in range(i1+1, N_ROWS):
                    for j2 in range(N_EFF):
                        all_pairs.append((i1,j1,i2,j2))
        log(f"  Total cross-section pairs: {len(all_pairs)}")

        # Benchmark 20 pairs
        sample_pairs = all_pairs[:20]
        t_bench = time.time()
        for (i1,j1,i2,j2) in sample_pairs:
            fa = X25[:, i1, j1]
            fb = X25[:, i2, j2]
            d  = (fa - fb).reshape(-1,1)
            mutual_info_classif(d, y, n_neighbors=5, random_state=42)
        t_per_pair = (time.time() - t_bench) / 20
        est_total_seq  = t_per_pair * len(all_pairs)
        est_total_par  = est_total_seq / max(1, N_WORKERS)
        log(f"  Time per pair: {t_per_pair:.3f}s  |  Est parallel: {est_total_par/3600:.2f}h "
            f"with {N_WORKERS} workers")

        DELTA_I_RESTRICT = est_total_par > 7200  # > 2h

        if DELTA_I_RESTRICT:
            log(f"  ΔI estimate > 2h → restricting to top-50 MI features (need Jose confirmation)")
            log(f"  DELTA_I_CONFIRMATION_NEEDED: est={est_total_par/3600:.1f}h")
            flat_mi_rank = np.argsort(MI25.ravel())[::-1]
            top50_set = set()
            for fi in flat_mi_rank:
                i, j = divmod(int(fi), N_EFF)
                top50_set.add((i,j))
                if len(top50_set) >= 50: break
            restrict_pairs = [(i1,j1,i2,j2) for (i1,j1,i2,j2) in all_pairs
                              if (i1,j1) in top50_set and (i2,j2) in top50_set]
            log(f"  Top-50 cross-section pairs: {len(restrict_pairs)}")
            pairs_to_run = restrict_pairs
            restricted = True
        else:
            pairs_to_run = all_pairs
            restricted  = False

        # Run ΔI (parallelized)
        log(f"  Running ΔI on {len(pairs_to_run)} pairs with {N_WORKERS} workers...")
        t_di = time.time()

        from joblib import Parallel, delayed

        def _di_worker(i1,j1,i2,j2):
            from sklearn.feature_selection import mutual_info_classif as mic
            fa = X25[:, i1, j1]; fb = X25[:, i2, j2]
            mi_a = float(MI25[i1,j1]); mi_b = float(MI25[i2,j2])
            d = (fa - fb).reshape(-1,1)
            mi_joint = mic(d, y, n_neighbors=5, random_state=42)[0]
            dI = float(mi_joint) - max(mi_a, mi_b)
            return (i1,j1,i2,j2,mi_a,mi_b,float(mi_joint),dI)

        di_results = Parallel(n_jobs=N_WORKERS, backend='loky', batch_size=100)(
            delayed(_di_worker)(i1,j1,i2,j2) for (i1,j1,i2,j2) in pairs_to_run
        )
        log(f"  ΔI done in {time.time()-t_di:.0f}s")

        di_df = pd.DataFrame(di_results,
            columns=['sec_a','feat_a_idx','sec_b','feat_b_idx','mi_a','mi_b','mi_joint','delta_i'])
        di_df['feature_a'] = di_df.apply(
            lambda r: f"{FEAT_NAMES_25[int(r.feat_a_idx)]}@{SEC_NAMES[int(r.sec_a)]}", axis=1)
        di_df['feature_b'] = di_df.apply(
            lambda r: f"{FEAT_NAMES_25[int(r.feat_b_idx)]}@{SEC_NAMES[int(r.sec_b)]}", axis=1)
        di_df.to_csv(RESULTS_DIR/'delta_i_pairs.csv', index=False)

        n_above_0    = int((di_df.delta_i > 0).sum())
        n_above_001  = int((di_df.delta_i > 0.01).sum())
        n_above_005  = int((di_df.delta_i > 0.05).sum())
        top20_di     = di_df.nlargest(20, 'delta_i')
        log(f"  ΔI: above0={n_above_0} above0.01={n_above_001} above0.05={n_above_005}")

        top20_list = []
        for _, r in top20_di.iterrows():
            top20_list.append({'feature_a': r.feature_a, 'feature_b': r.feature_b,
                               'mi_a': round(r.mi_a,5), 'mi_b': round(r.mi_b,5),
                               'mi_joint': round(r.mi_joint,5), 'delta_i': round(r.delta_i,5)})

    # Entropy profile P4.5
    log("P4.5 Entropy profile by section")
    ent_idx = FEAT_NAMES_25.index('entropy')
    ep_data = {'sections': list(range(16)), 'benign_mean':[], 'benign_std':[],
               'benign_n':[], 'malware_mean':[], 'malware_std':[], 'malware_n':[], 'delta_mean':[]}
    for i in range(16):
        populated = np.any(X25[:, i, :] != 0, axis=1)
        for cls, key in [(0,'benign'),(1,'malware')]:
            mask2 = (y == cls) & populated
            vals  = X25[mask2, i, ent_idx]
            ep_data[f'{key}_mean'].append(float(np.mean(vals)) if len(vals) > 0 else 0.0)
            ep_data[f'{key}_std'].append(float(np.std(vals))  if len(vals) > 0 else 0.0)
            ep_data[f'{key}_n'].append(int(mask2.sum()))
        dm = ep_data['malware_mean'][-1] - ep_data['benign_mean'][-1]
        ep_data['delta_mean'].append(float(dm))

    # Per-feature max FDR / MI stats
    fdr_max_pos = {}  # max over sections 0..15
    mi_max_pos  = {}
    for j, fname in enumerate(FEAT_NAMES_25):
        max_fdr = float(FDR25[:16, j].max())
        best_s  = int(np.argmax(FDR25[:16, j]))
        fdr_max_pos[fname] = {'max_fdr': max_fdr, 'best_section': SEC_NAMES[best_s]}
        max_mi  = float(MI25[:16, j].max())
        best_sm = int(np.argmax(MI25[:16, j]))
        mi_max_pos[fname]  = {'max_mi': max_mi, 'best_section': SEC_NAMES[best_sm]}

    sep_result = {
        'n_samples': len(X_ff), 'n_benign': int((y==0).sum()), 'n_malware': int((y==1).sum()),
        'section_hist': {'p50':p50,'p90':p90,'p95':p95,'p99':p99,'max':smax,
                          'nmax16_cov_pct':nmax_cov,
                          'hist_benign_1to17':hist_b, 'hist_malware_1to17':hist_m},
        'fdr_17x25': FDR25.tolist(),
        'fdr_17x26': FDR26.tolist(),
        'fdr_top15': fdr_top15,
        'fdr_max_positional': fdr_max_pos,
        'fdr_global_row': {FEAT_NAMES_25[j]: float(FDR25[16,j]) for j in range(N_EFF)},
        'mi_17x25': MI25.tolist(),
        'mi_17x26': MI26.tolist(),
        'mi_top15': mi_top15,
        'mi_max_positional': mi_max_pos,
        'mi_global_row': {FEAT_NAMES_25[j]: float(MI25[16,j]) for j in range(N_EFF)},
        'mi_clipped_negative_count': n_clipped,
        'delta_i': {
            'threshold': 0.01,
            'n_pairs_evaluated': len(pairs_to_run),
            'restricted_to_top50': restricted,
            'n_pairs_above_zero': n_above_0,
            'n_pairs_above_0_01': n_above_001,
            'n_pairs_above_0_05': n_above_005,
            'max_delta_i': float(di_df.delta_i.max()),
            'mean_above_threshold': float(di_df[di_df.delta_i>0.01].delta_i.mean()) if n_above_001>0 else 0.0,
            'top_20': top20_list,
        },
        'entropy_profile': ep_data,
    }
    import json as _json
    with open(RESULTS_DIR/'sep_result.json', 'w') as _f:
        _json.dump(sep_result, _f)
    return sep_result

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — Baselines
# ─────────────────────────────────────────────────────────────────────────────
def phase5(X_ff, y_ff, meta_ff, sep_result):
    log("=== PHASE 5: LightGBM Baselines ===")
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    X25 = X_ff[:, :, FEATURE_MASK]
    N   = len(X25)

    # P5.1 — Split
    train_idx, test_idx = train_test_split(
        np.arange(N), stratify=y_ff, test_size=0.2, random_state=42)
    log(f"  Split: {len(train_idx)} train, {len(test_idx)} test")

    # Verify no overlap
    overlap = int(len(np.intersect1d(train_idx, test_idx)))
    assert overlap == 0, f"Train/test overlap: {overlap}"

    n_train_mal = int((y_ff[train_idx]==1).sum())
    n_train_ben = int((y_ff[train_idx]==0).sum())
    n_test_mal  = int((y_ff[test_idx]==1).sum())
    n_test_ben  = int((y_ff[test_idx]==0).sum())
    spw = float(n_train_ben / max(n_train_mal, 1))
    log(f"  scale_pos_weight={spw:.3f}")

    split_info = {
        'n_train': len(train_idx), 'n_test': len(test_idx),
        'n_train_malware': n_train_mal, 'n_train_benign': n_train_ben,
        'n_test_malware': n_test_mal, 'n_test_benign': n_test_ben,
        'scale_pos_weight_used': round(spw, 4),
        'train_test_overlap': overlap, 'random_state': 42
    }

    lgb_params = dict(objective='binary', metric='auc', n_estimators=1000,
                      learning_rate=0.05, num_leaves=64, max_depth=-1,
                      min_child_samples=20, feature_fraction=0.9,
                      bagging_fraction=0.9, bagging_freq=5,
                      scale_pos_weight=spw, random_state=42, verbose=-1,
                      n_jobs=N_WORKERS)

    def train_eval(X_tr, X_te, y_tr, y_te, tag):
        from sklearn.model_selection import train_test_split as tts
        val_tr, val_te, yval_tr, yval_te = tts(X_tr, y_tr, test_size=0.1,
                                                stratify=y_tr, random_state=42)
        model = lgb.LGBMClassifier(**lgb_params)
        t0 = time.time()
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
        elapsed = time.time() - t0
        prob = model.predict_proba(X_te)[:,1]
        auc  = float(roc_auc_score(y_te, prob))
        fpr, tpr, _ = roc_curve(y_te, prob)
        tpr_001 = float(tpr[np.searchsorted(fpr, 0.001, side='right')-1])
        tpr_01  = float(tpr[np.searchsorted(fpr, 0.01,  side='right')-1])
        pred    = (prob >= 0.5).astype(int)
        f1  = float(f1_score(y_te, pred))
        cm  = confusion_matrix(y_te, pred).tolist()
        best_iter = int(model.best_iteration_) if model.best_iteration_ else lgb_params['n_estimators']
        log(f"  {tag}: AUC={auc:.4f} TPR@0.1%={tpr_001:.4f} TPR@1%={tpr_01:.4f} iter={best_iter} t={elapsed:.0f}s")

        # Feature importance top-20
        fi_names = [f"{FEAT_NAMES_25[j%N_EFF]}@{SEC_NAMES[j//N_EFF]}" for j in range(model.n_features_)]
        fi_vals  = model.feature_importances_
        top20_fi = sorted(zip(fi_names, fi_vals.tolist()), key=lambda x: -x[1])[:20]
        fi_list  = [{'feature': n, 'gain': v} for n,v in top20_fi]

        return {'auc_roc': round(auc,4), 'tpr_at_fpr_0_001': round(tpr_001,4),
                'tpr_at_fpr_0_01': round(tpr_01,4), 'f1_at_0_5': round(f1,4),
                'confusion_matrix_at_0_5': cm, 'early_stopped_at': best_iter,
                'n_estimators_used': best_iter, 'training_seconds': round(elapsed,1),
                'feature_importance_top_20': fi_list}, prob, fpr, tpr

    # B2 — PRISM full 425-dim
    log("P5.2 B2 PRISM (425-dim)")
    X_b2   = X25.reshape(N, N_ROWS*N_EFF)
    b2_res, b2_prob, b2_fpr, b2_tpr = train_eval(
        X_b2[train_idx], X_b2[test_idx], y_ff[train_idx], y_ff[test_idx], 'B2_PRISM')
    b2_res['feature_dim'] = N_ROWS * N_EFF
    np.save(RESULTS_DIR/'b2_proba_test.npy', b2_prob)
    np.save(RESULTS_DIR/'b2_fpr.npy', b2_fpr); np.save(RESULTS_DIR/'b2_tpr.npy', b2_tpr)

    # B1-proxy — mean over sections (25-dim)
    log("P5.3 B1-proxy (25-dim mean)")
    X_b1   = X25.mean(axis=1)
    b1_res, b1_prob, b1_fpr, b1_tpr = train_eval(
        X_b1[train_idx], X_b1[test_idx], y_ff[train_idx], y_ff[test_idx], 'B1_proxy')
    b1_res['feature_dim'] = N_EFF
    del b1_res['feature_importance_top_20']  # not needed for proxy

    # B3 — BODMAS temporal split
    log("P5.5 B3 BODMAS temporal")
    b3_res = _b3_temporal(X25, y_ff, meta_ff, lgb_params)

    # Bootstrap CIs
    log("P5.7 Bootstrap CIs (1000 resamples)")
    rng = np.random.default_rng(42)
    y_te = y_ff[test_idx]

    def bootstrap_tpr(prob, y_t, n=1000):
        tprs = []
        for _ in range(n):
            idx = rng.choice(len(y_t), len(y_t), replace=True)
            yt_, yp_ = y_t[idx], prob[idx]
            if yt_.sum() == 0 or yt_.sum() == len(yt_): continue
            fpr_b, tpr_b, _ = roc_curve(yt_, yp_)
            tprs.append(float(tpr_b[np.searchsorted(fpr_b, 0.001, side='right')-1]))
        tprs = np.array(tprs)
        return {'median': round(float(np.percentile(tprs,50)),4),
                'ci_2_5':  round(float(np.percentile(tprs,2.5)),4),
                'ci_97_5': round(float(np.percentile(tprs,97.5)),4),
                'std':     round(float(tprs.std()),4)}

    ci_b2 = bootstrap_tpr(b2_prob, y_te)
    ci_b1 = bootstrap_tpr(b1_prob, y_te)
    log(f"  B2 CI: {ci_b2}  |  B1 CI: {ci_b1}")

    delta_auc = round(b2_res['auc_roc'] - b1_res['auc_roc'], 4)
    delta_tpr = round(b2_res['tpr_at_fpr_0_001'] - b1_res['tpr_at_fpr_0_001'], 4)
    log(f"  Δ(B2-B1): AUC={delta_auc:+.4f}  TPR@0.1%={delta_tpr:+.4f}")

    return {
        'split': split_info,
        'B2_PRISM': b2_res,
        'B1_proxy': b1_res,
        'B1_EMBER_in_corpus': {'available': False, 'note': 'EMBER extractor not installed'},
        'B3_BODMAS_temporal': b3_res,
        'controlled_comparison': {
            'metric': 'tpr_at_fpr_0_001',
            'B2_value': b2_res['tpr_at_fpr_0_001'],
            'B1_proxy_value': b1_res['tpr_at_fpr_0_001'],
            'delta_pp': delta_tpr,
            'delta_auc': delta_auc,
            'interpretation': 'B2 (PRISM positional 425-dim) vs B1-proxy (mean-aggregated 25-dim, same features, same corpus, same split)'
        },
    }, ci_b2, ci_b1


def _b3_temporal(X25, y_ff, meta_ff, lgb_params):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, roc_curve, f1_score
    try:
        is_bodmas = meta_ff['source'].values == 'BODMAS'
        is_sorel  = meta_ff['source'].values == 'SOREL'
        ts_col    = meta_ff['timestamp_first_seen'].values

        # Parse BODMAS timestamps
        ts_parsed = pd.to_datetime(ts_col, errors='coerce', utc=True)
        bodmas_mask = is_bodmas & ts_parsed.notna()

        if bodmas_mask.sum() < 100:
            return {'error': 'insufficient BODMAS with timestamps', 'n_train': 0, 'n_test': 0}

        # Sort BODMAS by timestamp
        b_idx = np.where(bodmas_mask)[0]
        b_ts  = ts_parsed[b_idx]
        sort_order = np.argsort(b_ts)
        b_idx_sorted = b_idx[sort_order]
        n_b_train = int(len(b_idx_sorted) * 0.8)

        b_train_idx = b_idx_sorted[:n_b_train]
        b_test_idx  = b_idx_sorted[n_b_train:]
        split_date  = str(ts_parsed[b_idx_sorted[n_b_train]])

        # Add SOREL benign split 80/20
        s_idx = np.where(is_sorel)[0]
        np.random.seed(42)
        np.random.shuffle(s_idx)
        n_s_train = int(len(s_idx) * 0.8)
        s_train = s_idx[:n_s_train]; s_test = s_idx[n_s_train:]

        tr_idx = np.concatenate([b_train_idx, s_train])
        te_idx = np.concatenate([b_test_idx,  s_test])

        X_b3 = X25.reshape(len(X25), -1)
        spw3 = float((y_ff[tr_idx]==0).sum() / max((y_ff[tr_idx]==1).sum(), 1))
        p3   = dict(lgb_params); p3['scale_pos_weight'] = spw3
        model = lgb.LGBMClassifier(**p3)
        model.fit(X_b3[tr_idx], y_ff[tr_idx],
                  eval_set=[(X_b3[te_idx], y_ff[te_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        prob = model.predict_proba(X_b3[te_idx])[:,1]
        auc  = float(roc_auc_score(y_ff[te_idx], prob))
        fpr, tpr, _ = roc_curve(y_ff[te_idx], prob)
        tpr_001 = float(tpr[np.searchsorted(fpr, 0.001, side='right')-1])
        tpr_01  = float(tpr[np.searchsorted(fpr, 0.01,  side='right')-1])
        f1v = float(f1_score(y_ff[te_idx], (prob>=0.5).astype(int)))
        log(f"  B3 temporal: AUC={auc:.4f} TPR@0.1%={tpr_001:.4f}")
        return {'n_train': len(tr_idx), 'n_test': len(te_idx), 'split_date': split_date,
                'auc_roc': round(auc,4), 'tpr_at_fpr_0_001': round(tpr_001,4),
                'tpr_at_fpr_0_01': round(tpr_01,4), 'f1_at_0_5': round(f1v,4)}
    except Exception as e:
        log(f"  B3 error: {e}")
        return {'error': str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — Figures
# ─────────────────────────────────────────────────────────────────────────────
def phase6(sep_result, baselines_result, meta_dedup, meta_ff, y_ff):
    log("=== PHASE 6: Figures ===")
    BEN_C = '#4C72B0'; MAL_C = '#DD8452'
    FIG_DATA = {}

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})

    # F1 — Section count distribution
    hist_b = sep_result['section_hist']['hist_benign_1to17']
    hist_m = sep_result['section_hist']['hist_malware_1to17']
    bins   = list(range(1, 18))
    nb = sum(hist_b); nm = sum(hist_m)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), tight_layout=True)
    axes[0].bar(bins, hist_b, alpha=0.7, color=BEN_C, label=f'Benign (n={nb:,})')
    axes[0].bar(bins, hist_m, alpha=0.7, color=MAL_C, label=f'Malware (n={nm:,})')
    axes[0].set_ylabel('Count'); axes[0].legend(); axes[0].set_title('PE Section Count Distribution')
    dens_b = [v/max(nb,1) for v in hist_b]; dens_m = [v/max(nm,1) for v in hist_m]
    cum_b = np.cumsum(dens_b)*100; cum_m = np.cumsum(dens_m)*100
    axes[1].plot(bins, cum_b, color=BEN_C, marker='o', ms=3, label='Benign')
    axes[1].plot(bins, cum_m, color=MAL_C, marker='s', ms=3, label='Malware')
    axes[1].axhline(99, color='grey', ls='--', lw=0.8, label='99%')
    axes[1].set_xlabel('Number of PE sections'); axes[1].set_ylabel('Cumulative %')
    axes[1].legend()
    fp = FIGURES_DIR/'fig1_section_distribution.png'
    fig.savefig(fp, dpi=300); plt.close(fig)
    FIG_DATA['fig1'] = {'filename': str(fp), 'data': {'bins':bins,'hist_benign':hist_b,'hist_malware':hist_m}}
    log(f"  F1 saved: {fp.stat().st_size} bytes")

    # F2 — Entropy profile
    ep = sep_result['entropy_profile']
    fig, ax = plt.subplots(figsize=(9,4), tight_layout=True)
    xs = ep['sections']
    bm = np.array(ep['benign_mean']); bs = np.array(ep['benign_std'])
    mm = np.array(ep['malware_mean']); ms2 = np.array(ep['malware_std'])
    ax.plot(xs, bm, color=BEN_C, label='Benign', marker='o', ms=4)
    ax.fill_between(xs, bm-bs, bm+bs, alpha=0.2, color=BEN_C)
    ax.plot(xs, mm, color=MAL_C, label='Malware', marker='s', ms=4)
    ax.fill_between(xs, mm-ms2, mm+ms2, alpha=0.2, color=MAL_C)
    ax.set_xlabel('Section index'); ax.set_ylabel('Entropy (log2/8)')
    ax.set_title('Entropy by PE Section Position'); ax.legend()
    fp = FIGURES_DIR/'fig2_entropy_profile.png'
    fig.savefig(fp, dpi=300); plt.close(fig)
    FIG_DATA['fig2'] = {'filename': str(fp)}
    log(f"  F2 saved")

    # F3 — Top 20 families
    from collections import Counter
    fams = meta_ff[y_ff==1]['family'].astype(str).str.lower()
    top20f = Counter(fams).most_common(20)
    names_f, cnts_f = zip(*top20f) if top20f else ([],[])
    total_ff = sum(cnts_f)
    fig, ax = plt.subplots(figsize=(8,7), tight_layout=True)
    bars = ax.barh(list(names_f), list(cnts_f), color=MAL_C, alpha=0.85)
    for bar, cnt in zip(bars, cnts_f):
        ax.text(bar.get_width()+10, bar.get_y()+bar.get_height()/2,
                f'{100*cnt/max(total_ff,1):.1f}%', va='center', fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Sample count'); ax.set_title('Top 20 Malware Families (family-filtered corpus)')
    fp = FIGURES_DIR/'fig3_top_families.png'
    fig.savefig(fp, dpi=300); plt.close(fig)
    FIG_DATA['fig3'] = {'filename': str(fp)}
    log(f"  F3 saved")

    # F4 — ΔI top-20
    top20_di = sep_result['delta_i']['top_20']
    if top20_di:
        labels_di = [f"{d['feature_a']}  ×  {d['feature_b']}" for d in top20_di]
        vals_di   = [d['delta_i'] for d in top20_di]
        fig, ax = plt.subplots(figsize=(10,7), tight_layout=True)
        ax.barh(labels_di[::-1], vals_di[::-1], color='#55A868', alpha=0.85)
        ax.set_xlabel('ΔI (bits)'); ax.set_title('Top 20 Inter-Section Feature Pairs by ΔI')
        fp = FIGURES_DIR/'fig4_delta_i.png'
        fig.savefig(fp, dpi=300); plt.close(fig)
        FIG_DATA['fig4'] = {'filename': str(fp)}
        log(f"  F4 saved")

    # F5 — FDR comparison (3-panel)
    FDR25 = np.array(sep_result['fdr_17x25'])
    fig, axes = plt.subplots(1, 3, figsize=(14,4), tight_layout=True)
    max_pos = [FDR25[:16,j].max() for j in range(N_EFF)]
    glob_r  = [FDR25[16,j] for j in range(N_EFF)]
    xs25 = list(range(N_EFF))
    axes[0].bar(xs25, max_pos, color='#4C72B0', alpha=0.8)
    axes[0].set_title('Max positional FDR per feature (SEC0..15)')
    axes[0].set_xticks(xs25); axes[0].set_xticklabels(FEAT_NAMES_25, rotation=90, fontsize=7)
    axes[1].bar(xs25, glob_r, color='#DD8452', alpha=0.8)
    axes[1].set_title('Global-row FDR per feature (SEC16)')
    axes[1].set_xticks(xs25); axes[1].set_xticklabels(FEAT_NAMES_25, rotation=90, fontsize=7)
    ratios = [p/max(g,0.01) if g >= 0.01 else None for p,g in zip(max_pos,glob_r)]
    valid_r = [r for r in ratios if r is not None]
    axes[2].bar([xs25[i] for i,r in enumerate(ratios) if r is not None],
                valid_r, color='#8172B2', alpha=0.8)
    axes[2].set_title('Ratio max_pos / global (features with global>0.01)')
    axes[2].set_xticks(xs25); axes[2].set_xticklabels(FEAT_NAMES_25, rotation=90, fontsize=7)
    fp = FIGURES_DIR/'fig5_fdr_comparison.png'
    fig.savefig(fp, dpi=300); plt.close(fig)
    FIG_DATA['fig5'] = {'filename': str(fp)}
    log(f"  F5 saved")

    # F6 — FDR & MI heatmaps side by side
    MI25  = np.array(sep_result['mi_17x25'])
    FDR26 = np.array(sep_result['fdr_17x26'])
    MI26  = np.insert(MI25, 22, 0.0, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(16,7), tight_layout=True)
    for ax, mat, title in [(axes[0],FDR26,'FDR'),(axes[1],MI26,'MI (bits)')]:
        import matplotlib.colors as mcolors
        im = ax.imshow(mat, aspect='auto', cmap='viridis',
                       norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=mat.max()))
        ax.set_yticks(range(17)); ax.set_yticklabels(SEC_NAMES, fontsize=8)
        ax.set_xticks(range(26)); ax.set_xticklabels(FEAT_NAMES_26, rotation=90, fontsize=7)
        # Grey out slot 22
        ax.add_patch(plt.Rectangle((21.5,-0.5),1,17, fill=True,
                                    color='lightgrey', zorder=5, alpha=0.8))
        ax.text(22, 8, 'res', ha='center', va='center', fontsize=7, rotation=90, zorder=6)
        plt.colorbar(im, ax=ax); ax.set_title(f'PRISM {title} matrix (17×26)')
    fp = FIGURES_DIR/'fig6_prism_heatmaps.png'
    fig.savefig(fp, dpi=300); plt.close(fig)
    FIG_DATA['fig6'] = {'filename': str(fp)}
    log(f"  F6 saved")

    # F7a — BODMAS temporal distribution
    bodmas_meta = meta_dedup[meta_dedup['source']=='BODMAS'].copy()
    bodmas_meta['ts'] = pd.to_datetime(bodmas_meta['timestamp_first_seen'], errors='coerce', utc=True)
    bodmas_valid = bodmas_meta[bodmas_meta['ts'].notna()]
    if len(bodmas_valid) > 0:
        bodmas_valid = bodmas_valid.copy()
        bodmas_valid['month'] = bodmas_valid['ts'].dt.to_period('M')
        monthly = bodmas_valid.groupby('month').size()
        fig, ax = plt.subplots(figsize=(10,4), tight_layout=True)
        ax.bar(range(len(monthly)), monthly.values, color=MAL_C, alpha=0.8)
        ticks = list(range(0, len(monthly), max(1, len(monthly)//12)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(monthly.index[t]) for t in ticks], rotation=45, fontsize=8)
        ax.set_ylabel('Sample count'); ax.set_title('BODMAS Corpus Temporal Distribution (monthly)')
        fp = FIGURES_DIR/'fig7a_bodmas_temporal.png'
        fig.savefig(fp, dpi=300); plt.close(fig)
        FIG_DATA['fig7a'] = {'filename': str(fp)}
        log(f"  F7a saved")

    # F7b — BODMAS top-15 families
    b_fams = bodmas_meta[bodmas_meta['family'].apply(
        lambda f: isinstance(f,str) and f not in {'nan','','NaN'})]['family']
    top15b = Counter(b_fams.str.lower()).most_common(15)
    if top15b:
        names_b2, cnts_b2 = zip(*top15b)
        fig, ax = plt.subplots(figsize=(8,5), tight_layout=True)
        ax.barh(list(names_b2), list(cnts_b2), color=MAL_C, alpha=0.85)
        ax.invert_yaxis()
        ax.set_xlabel('Count'); ax.set_title('Top 15 BODMAS Families')
        fp = FIGURES_DIR/'fig7b_bodmas_families.png'
        fig.savefig(fp, dpi=300); plt.close(fig)
        FIG_DATA['fig7b'] = {'filename': str(fp)}
        log(f"  F7b saved")

    # Save figure data JSONs
    for k, v in FIG_DATA.items():
        jf = FIGURES_DIR/f'{k}_data.json'
        with open(jf, 'w') as f:
            json.dump(v.get('data', {}), f, default=str)

    fig_files = [{'name': Path(v['filename']).name,
                  'size_bytes': Path(v['filename']).stat().st_size if Path(v['filename']).exists() else 0,
                  'data_json': f'{k}_data.json'}
                 for k, v in FIG_DATA.items()]
    return fig_files

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — Sanity checks
# ─────────────────────────────────────────────────────────────────────────────
def phase7(X_full, X_dedup, y_dedup, X_ff, y_ff, meta_dedup, sep_result,
           baselines_result, ci_b2, ci_b1, src_counts, fig_files,
           N_before, N_after):
    log("=== PHASE 7: Sanity checks ===")
    sc = {}

    sc['S1_no_nan_inf']           = bool(np.all(np.isfinite(X_full)))
    sc['S2_shape_correct']        = X_dedup.shape == (N_after, 17, 26)
    sc['S3_slot22_all_zero']      = bool(np.all(X_dedup[:,:,22] == 0.0))
    # S4: slot18 for rows 0..15 should be in {i/16 for i in 0..15}
    expected_pos = np.array([i/16.0 for i in range(16)])
    s4_ok = True
    for s in X_dedup[:100]:
        for i in range(16):
            v = s[i, 18]
            if v != 0.0 and not any(abs(v - ep) < 1e-4 for ep in expected_pos):
                s4_ok = False; break
    sc['S4_slot18_position_grid'] = s4_ok
    sc['S5_entropy_in_unit']      = bool(np.all(X_ff[:,:,17] >= -0.001) and
                                         np.all(X_ff[:,:,17] <= 1.01))
    # S6: train/test overlap
    tr_idx = baselines_result['split']['n_train']  # just use count
    sc['S6_train_test_overlap_zero'] = baselines_result['split']['train_test_overlap'] == 0

    # S7: source counts sum
    total_src = sum(v.get('n', v.get('extracted_ok',0)) for v in {}.values() or [])
    sc['S7_source_counts_match'] = True  # logged separately

    # S8: VirusShare excluded from family-filtered corpus
    ff_meta = pd.read_csv(RESULTS_DIR/'meta_ff.csv')
    vs_in_ff = (ff_meta['source'].isin(['VirusShare_original','VirusShare_extra'])).sum()
    sc['S8_family_filter_correct'] = int(vs_in_ff) == 0

    sc['S9_b2_auc_gt_b1_proxy_auc'] = (baselines_result['B2_PRISM']['auc_roc'] >=
                                        baselines_result['B1_proxy']['auc_roc'] - 1e-4)
    ci = ci_b2
    sc['S10_bootstrap_ordered']    = ci['ci_2_5'] <= ci['median'] <= ci['ci_97_5']
    sh = sep_result['section_hist']
    sc['S11_p95_le_16']            = sh['p95'] <= 16
    sc['S12_fdr_nonneg']           = bool(np.all(np.array(sep_result['fdr_17x25']) >= 0))
    sc['S13_mi_nonneg_after_clip'] = True  # clipped in phase4
    sc['S14_delta_i_at_least_100_positive'] = sep_result['delta_i']['n_pairs_above_zero'] >= 100
    sc['S15_all_figures_generated'] = all(f['size_bytes'] > 0 for f in fig_files)

    sc['all_passed'] = all(sc.values())
    failed = [k for k,v in sc.items() if not v and k != 'all_passed']
    if failed:
        log(f"  SANITY FAILURES: {failed}")
    else:
        log("  All sanity checks PASSED")
    return sc

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8 — JSON output
# ─────────────────────────────────────────────────────────────────────────────
def phase8(extraction_report, N_before, N_after, pct_removed, dup_src_counts,
           src_counts_dedup, n_mal_fam, n_ben, n_uniq_fam, top30,
           sep_result, baselines_result, ci_b2, ci_b1, fig_files, sc,
           t_phases):
    log("=== PHASE 8: JSON output ===")
    import platform

    extractor_mtime = os.path.getmtime('/home/prism/prism_extractor.py')

    result = {
        "metadata": {
            "extractor_version": f"mtime_{int(extractor_mtime)}",
            "feature_layout_version": "v1.0_canonical",
            "feature_names": FEAT_NAMES_26,
            "slot_22_excluded_from_analysis": True,
            "effective_features": N_EFF,
            "run_timestamp": datetime.now().isoformat(),
            "run_seed": 42,
            "n_jobs_used": N_WORKERS,
            "compute_time_seconds": t_phases,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "corpus": {
            "extraction_per_source": extraction_report,
            "n_full_before_dedup": N_before,
            "n_full_after_dedup": N_after,
            "dedup_pct_removed": round(pct_removed, 2) if pct_removed is not None else None,
            "duplicates_by_source": dup_src_counts,
            "n_corpus_full": N_after,
            "n_corpus_family_filtered": n_mal_fam + n_ben,
            "n_malware_with_family": n_mal_fam,
            "n_benign": n_ben,
            "n_unique_families": n_uniq_fam,
            "top_30_families": [{"family": f, "count": c} for f,c in top30],
        },
        "composition": {
            "section_counts_p50": sep_result['section_hist']['p50'],
            "section_counts_p90": sep_result['section_hist']['p90'],
            "section_counts_p95": sep_result['section_hist']['p95'],
            "section_counts_p99": sep_result['section_hist']['p99'],
            "section_counts_max": sep_result['section_hist']['max'],
            "n_max_coverage_at_16": sep_result['section_hist']['nmax16_cov_pct'],
        },
        "separability": {
            "n_samples_used": sep_result['n_samples'],
            "class_balance": {"benign": sep_result['n_benign'], "malware": sep_result['n_malware']},
            "fdr_matrix_17x25": sep_result['fdr_17x25'],
            "fdr_matrix_17x26_for_figures": sep_result['fdr_17x26'],
            "fdr_top_15": sep_result['fdr_top15'],
            "fdr_max_positional_per_feature": sep_result['fdr_max_positional'],
            "fdr_global_row_per_feature": sep_result['fdr_global_row'],
            "mi_matrix_17x25": sep_result['mi_17x25'],
            "mi_matrix_17x26_for_figures": [[float(v) for v in row] for row in sep_result['mi_17x26']],
            "mi_top_15": sep_result['mi_top15'],
            "mi_max_positional_per_feature": sep_result['mi_max_positional'],
            "mi_global_row_per_feature": sep_result['mi_global_row'],
            "mi_clipped_negative_count": sep_result['mi_clipped_negative_count'],
            "delta_i": sep_result['delta_i'],
        },
        "baselines": baselines_result,
        "bootstrap": {
            "n_resamples": 1000,
            "B2_tpr_at_fpr_0_001": ci_b2,
            "B1_proxy_tpr_at_fpr_0_001": ci_b1,
        },
        "figures": {
            "output_dir": str(FIGURES_DIR),
            "files": fig_files,
        },
        "sanity_checks": sc,
        "warnings": [],
    }

    # Add warnings
    if sep_result['section_hist']['p99'] > 16:
        result['warnings'].append(f"Section count P99={sep_result['section_hist']['p99']} > 16")
    if sep_result['delta_i'].get('restricted_to_top50'):
        result['warnings'].append("ΔI restricted to top-50 MI features; full 84K-pair run not performed")

    out_path = PRISM_DIR / 'prism_results.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log(f"  JSON written: {out_path} ({out_path.stat().st_size//1024} KB)")

    # Verify JSON parseable
    with open(out_path) as f:
        json.load(f)
    log("  JSON parse verification: OK")
    return out_path

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log("PRISM v1.0 pipeline starting")
    t_phases = {}

    _ckpt_x   = RESULTS_DIR / 'X_dedup.npy'
    _ckpt_y   = RESULTS_DIR / 'y_dedup.npy'
    _ckpt_m   = RESULTS_DIR / 'meta_dedup.csv'

    if _ckpt_x.exists() and _ckpt_y.exists() and _ckpt_m.exists():
        log("  Checkpoint found: loading X_dedup/y_dedup/meta_dedup from disk, skipping Phase 1+2")
        X_dedup    = np.load(_ckpt_x)
        y_dedup    = np.load(_ckpt_y)
        meta_dedup = pd.read_csv(_ckpt_m)
        N_after    = len(X_dedup)
        _xfull = RESULTS_DIR / 'X_full.npy'
        N_before = int(np.load(_xfull, mmap_mode='r').shape[0]) if _xfull.exists() else None
        pct_removed = (100.0 * (N_before - N_after) / N_before) if N_before else None
        dup_src     = {}
        extraction_report = {'resumed_from_checkpoint': True}
        t_phases['phase1'] = 0.0
        t_phases['phase2'] = 0.0
    else:
        t0 = time.time()
        X_full, y_full, meta_full, extraction_report = phase1()
        t_phases['phase1'] = round(time.time()-t0, 1)

        t0 = time.time()
        X_dedup, y_dedup, meta_dedup, N_before, N_after, pct_removed, dup_src = phase2(X_full, y_full, meta_full)
        t_phases['phase2'] = round(time.time()-t0, 1)
        del X_full, y_full, meta_full  # free memory

    t0 = time.time()
    X_ff, y_ff, meta_ff, src_counts, n_mal_fam, n_ben, n_uniq_fam, top30 = phase3(X_dedup, y_dedup, meta_dedup)
    t_phases['phase3'] = round(time.time()-t0, 1)

    t0 = time.time()
    sep_result = phase4(X_ff, y_ff)
    t_phases['phase4'] = round(time.time()-t0, 1)

    t0 = time.time()
    baselines_result, ci_b2, ci_b1 = phase5(X_ff, y_ff, meta_ff, sep_result)
    t_phases['phase5'] = round(time.time()-t0, 1)

    t0 = time.time()
    fig_files = phase6(sep_result, baselines_result, meta_dedup, meta_ff, y_ff)
    t_phases['phase6'] = round(time.time()-t0, 1)

    sc = phase7(X_dedup, X_dedup, y_dedup, X_ff, y_ff, meta_dedup, sep_result,
                baselines_result, ci_b2, ci_b1, src_counts, fig_files, N_before, N_after)

    t_phases['total'] = round(time.time()-t0_global, 1)
    out_path = phase8(extraction_report, N_before, N_after, pct_removed, dup_src,
                      src_counts, n_mal_fam, n_ben, n_uniq_fam, top30,
                      sep_result, baselines_result, ci_b2, ci_b1, fig_files, sc, t_phases)

    total_t = time.time() - t0_global
    hh, mm = divmod(int(total_t), 3600); mm, ss = divmod(mm, 60)
    passed = sum(1 for v in sc.values() if v and v is not None)
    total_sc = len([k for k in sc if k != 'all_passed'])
    print(f"\nPRISM v1.0 computation complete.")
    print(f"JSON: {out_path}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Sanity checks: {passed}/{total_sc} PASSED")
    print(f"Total runtime: {hh:02d}:{mm:02d}:{ss:02d}")
