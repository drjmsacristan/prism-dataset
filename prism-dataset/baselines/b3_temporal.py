"""
B3-BODMAS temporal split baseline.

Calls _b3_temporal() in analysis/prism_pipeline.py.
The B3 model trains on BODMAS 2019-08 to 2020-04 and tests on 2020-05 to 2020-09.
Temporal split applies only to malware samples (benigns have no timestamps).

Paper result (family-filtered corpus, BODMAS temporal):
    AUC = 0.9997   TPR@FPR=0.1% = 0.9969
"""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from prism_pipeline import _b3_temporal, FEATURE_MASK


def run_b3(X_ff_path, y_ff_path, meta_ff_path):
    """
    Re-runs B3 BODMAS temporal split from saved .npy arrays.
    Args:
        X_ff_path: path to X_ff.npy  (N, 17, 26) float32
        y_ff_path: path to y_ff.npy  (N,) uint8
        meta_ff_path: path to meta_ff.csv (needs 'timestamp' and 'source' columns)
    """
    import pandas as pd
    X_ff = np.load(X_ff_path)
    y_ff = np.load(y_ff_path)
    meta_ff = pd.read_csv(meta_ff_path)

    X25 = X_ff[:, :, FEATURE_MASK]
    lgb_params = dict(objective='binary', metric='auc', n_estimators=1000,
                      learning_rate=0.05, num_leaves=64, max_depth=-1,
                      min_child_samples=20, feature_fraction=0.9,
                      bagging_fraction=0.9, bagging_freq=5,
                      random_state=42, verbose=-1, n_jobs=1,
                      num_threads=1, force_row_wise=True, deterministic=True)
    return _b3_temporal(X25, y_ff, meta_ff, lgb_params)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python run_b3_temporal.py <X_ff.npy> <y_ff.npy> <meta_ff.csv>")
        sys.exit(1)
    import json
    res = run_b3(sys.argv[1], sys.argv[2], sys.argv[3])
    print(json.dumps(res, indent=2))
