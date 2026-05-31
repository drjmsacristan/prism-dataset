"""
B1-proxy baseline: mean-aggregated PRISM (25-dim).

Calls phase5() in analysis/prism_pipeline.py.
The B1-proxy model is trained inside phase5() as 'B1_proxy':
    X_b1 = X25.mean(axis=1)          # shape (N, 25) — mean over 17 sections
    model trained with LightGBM (lgb_params), same split as B2.

To run the full baseline suite (B1 + B2 + B3) execute prism_pipeline.py directly.
This wrapper shows how to call phase5() in isolation given pre-computed arrays.
"""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from prism_pipeline import phase5, FEATURE_MASK, RESULTS_DIR

RESULTS = Path(__file__).resolve().parents[1] / "results"


def run_b1_proxy(X_ff_path, y_ff_path, meta_ff_path, sep_result=None):
    """
    Re-runs B1-proxy from saved .npy arrays.
    Args:
        X_ff_path: path to X_ff.npy  (N, 17, 26) float32
        y_ff_path: path to y_ff.npy  (N,) uint8
        meta_ff_path: path to meta_ff.csv
        sep_result: optional separability dict (pass None to skip checks)
    Returns:
        dict with keys 'B1_proxy', 'B2_PRISM', 'B3_BODMAS_temporal'
    """
    import pandas as pd
    X_ff = np.load(X_ff_path)
    y_ff = np.load(y_ff_path)
    meta_ff = pd.read_csv(meta_ff_path)
    result = phase5(X_ff, y_ff, meta_ff, sep_result or {})
    return result


if __name__ == "__main__":
    # Example: python run_b1_proxy.py X_ff.npy y_ff.npy meta_ff.csv
    if len(sys.argv) < 4:
        print("Usage: python run_b1_proxy.py <X_ff.npy> <y_ff.npy> <meta_ff.csv>")
        sys.exit(1)
    res = run_b1_proxy(sys.argv[1], sys.argv[2], sys.argv[3])
    import json
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'feature_importance_top_20'}
                      for k, v in res.items() if isinstance(v, dict) and 'auc_roc' in v}, indent=2))
