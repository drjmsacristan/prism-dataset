#!/usr/bin/env python3
"""
ember_extract_bodmas.py
=======================
Extract EMBER v2 (2,381-dim) feature vectors from restored BODMAS binaries,
for the controlled cross-representation benchmark (baseline B4).

The EMBER v2 feature extractor was developed against LIEF 0.9 and NumPy < 1.24.
This project runs LIEF 0.14.1 and NumPy >= 1.24, so two runtime compatibility
patches are applied before importing `ember` (these are the patches referenced
in the paper, Section VI-E footnote):

  (a) NumPy type-alias patch: restore the removed aliases np.int / np.float /
      np.bool / np.object / np.str / np.complex as their builtin equivalents.
  (b) LIEF exception-class patch: alias the exception classes that LIEF 0.9
      exposed (bad_format, bad_file, pe_error, parser_error, read_out_of_bound)
      to the generic Exception, since LIEF 0.14 reorganised them.

A residual gap remains: `lief.not_found` is not patched, which is the documented
cause of the ~1% extraction-failure rate on the non-BODMAS sub-corpus. On the
restored BODMAS binaries, extraction succeeds for 57,033 / 57,058 (99.96%).

INPUT  : a directory of *restored* BODMAS binaries (see bodmas_restore.py) and a
         text file listing the sample order (one filename per line) so that the
         output rows align with the PRISM matrices.
OUTPUT : X_bodmas_ember.npy  (n_ok, 2381) float32
         bodmas_ember_idx.npy (n_ok,)      int   -> row index into the name list
         bodmas_ember_summary.json

SECURITY NOTE
-------------
Operates on functional malware binaries; run only inside an isolated VM. No
binaries are distributed with PRISM.

Usage
-----
    python ember_extract_bodmas.py \
        --bindir /path/restored \
        --names  nombres_bodmas.txt \
        --outdir /path/prism_matrices \
        --workers 10
"""

import argparse
import builtins
import json
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Compatibility patches — MUST run before `import ember`.
# ---------------------------------------------------------------------------
def apply_numpy_aliases():
    """(a) Restore NumPy scalar aliases removed in NumPy 1.24+."""
    aliases = {
        "int": int, "float": float, "bool": bool,
        "object": object, "str": str, "complex": complex,
    }
    for name, target in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, target)


def apply_lief_exception_aliases():
    """(b) Alias LIEF 0.9 exception classes to Exception under LIEF 0.14."""
    import lief
    for name in ("bad_format", "bad_file", "pe_error",
                 "parser_error", "read_out_of_bound"):
        if not hasattr(lief, name):
            setattr(lief, name, Exception)


def make_extractor():
    apply_numpy_aliases()
    apply_lief_exception_aliases()
    import ember  # imported only after patches are in place
    return ember.PEFeatureExtractor(feature_version=2)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
_EXTRACTOR = None  # per-process singleton


def _init_worker():
    global _EXTRACTOR
    _EXTRACTOR = make_extractor()


def _extract_one(args):
    """Return (row_index, vector | None)."""
    row_index, path = args
    try:
        data = Path(path).read_bytes()
        vec = np.asarray(_EXTRACTOR.feature_vector(data), dtype=np.float32)
        if vec.shape[0] != 2381 or not np.all(np.isfinite(vec)):
            return row_index, None
        return row_index, vec
    except Exception:  # noqa: BLE001 - count as failure, keep going
        return row_index, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="EMBER v2 extraction over restored BODMAS.")
    ap.add_argument("--bindir", required=True, help="directory of restored binaries")
    ap.add_argument("--names", required=True,
                    help="text file: one binary filename per line, in matrix order")
    ap.add_argument("--outdir", required=True, help="output directory for .npy/.json")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    bindir = Path(args.bindir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    names = [ln.strip() for ln in Path(args.names).read_text().splitlines() if ln.strip()]
    # Build (row_index, path) only for names whose binary exists on disk.
    tasks = []
    for i, name in enumerate(names):
        p = bindir / name
        if p.is_file():
            tasks.append((i, str(p)))
    print(f"Candidates with a binary on disk: {len(tasks)} / {len(names)}", flush=True)

    t0 = time.time()
    results = []
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers, initializer=_init_worker) as pool:
            for k, r in enumerate(pool.imap_unordered(_extract_one, tasks, chunksize=64), 1):
                results.append(r)
                if k % 2000 == 0:
                    rate = k / (time.time() - t0)
                    print(f"  [{k}/{len(tasks)}] rate={rate:.0f}/s", flush=True)
    else:
        _init_worker()
        for k, t in enumerate(tasks, 1):
            results.append(_extract_one(t))
            if k % 2000 == 0:
                print(f"  [{k}/{len(tasks)}]", flush=True)

    ok = [(idx, vec) for idx, vec in results if vec is not None]
    ok.sort(key=lambda r: r[0])
    n_ok, n_fail = len(ok), len(tasks) - len(ok)

    if ok:
        idx = np.array([i for i, _ in ok], dtype=np.int64)
        X = np.stack([v for _, v in ok]).astype(np.float32)
    else:
        idx = np.zeros((0,), dtype=np.int64)
        X = np.zeros((0, 2381), dtype=np.float32)

    np.save(outdir / "X_bodmas_ember.npy", X)
    np.save(outdir / "bodmas_ember_idx.npy", idx)
    summary = {
        "candidates": len(tasks),
        "ok": n_ok,
        "fail": n_fail,
        "success_pct": round(100.0 * n_ok / max(len(tasks), 1), 2),
        "elapsed_min": round((time.time() - t0) / 60.0, 1),
        "shape": list(X.shape),
    }
    (outdir / "bodmas_ember_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved X_bodmas_ember.npy {X.shape} and bodmas_ember_idx.npy -> {outdir}")


if __name__ == "__main__":
    main()
