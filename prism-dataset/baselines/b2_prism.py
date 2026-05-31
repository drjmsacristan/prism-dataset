"""
B2-PRISM baseline: full positional PRISM (17 × 25 = 425-dim).

Calls phase5() in analysis/prism_pipeline.py.
The B2 model is trained inside phase5() as 'B2_PRISM':
    X_b2 = X25.reshape(N, N_ROWS * N_EFF)  # shape (N, 425)
    model trained with LightGBM (lgb_params), stratified 80/20 split, seed=42.

Paper result (seed 42, family-filtered corpus 49.204):
    AUC = 0.9998   TPR@FPR=0.1% = 0.9924

See run_b1_proxy.py for the unified phase5() call that runs B1, B2, B3 together.
"""
# B2 is produced by the same phase5() call as B1. See run_b1_proxy.py.
# Kept as a separate file for documentation clarity.
raise NotImplementedError(
    "Run run_b1_proxy.py — B2 is computed inside the same phase5() call."
)
