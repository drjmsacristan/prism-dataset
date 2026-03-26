import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
import time

print('Cargando vectores BODMAS completos (malware + benigno)...')
data = np.load('/home/prism/data/bodmas.npz')
X_bod = data['X']
y_bod = data['y']

meta = pd.read_csv('/home/prism/data/bodmas_metadata.csv')
meta.columns = ['sha', 'timestamp', 'family']
meta['timestamp'] = pd.to_datetime(meta['timestamp'], format='mixed', utc=True)
meta['year'] = meta['timestamp'].dt.year

print(f'Total: {len(y_bod):,} | Malware: {(y_bod==1).sum():,} | Benigno: {(y_bod==0).sum():,}')

# Split temporal justo: train=2019, test=2020
idx_2019 = meta[meta['year'] == 2019].index
idx_2020 = meta[meta['year'] == 2020].index

# Filtrar indices validos
idx_2019 = idx_2019[idx_2019 < len(X_bod)]
idx_2020 = idx_2020[idx_2020 < len(X_bod)]

X_train = X_bod[idx_2019]
y_train = y_bod[idx_2019]
X_test  = X_bod[idx_2020]
y_test  = y_bod[idx_2020]

print(f'\nTrain 2019: {len(y_train):,} | Mal: {y_train.sum():,} | Ben: {(y_train==0).sum():,}')
print(f'Test  2020: {len(y_test):,}  | Mal: {y_test.sum():,} | Ben: {(y_test==0).sum():,}')

# Entrenar con vector EMBER 1D (B1 sobre BODMAS)
print('\nEntrenando B1-BODMAS (EMBER 1D, split temporal 2019→2020)...')
ratio = (y_train==0).sum() / max((y_train==1).sum(), 1)
t0 = time.time()
m_b1 = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    min_child_samples=20, n_jobs=8, random_state=42,
    scale_pos_weight=ratio)
m_b1.fit(X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(200)])
t1 = time.time()

yp = m_b1.predict_proba(X_test)[:,1]
auc_b1 = roc_auc_score(y_test, yp)
fpr, tpr, _ = roc_curve(y_test, yp)
tpr_01_b1 = tpr[np.searchsorted(fpr, 0.001)]
tpr_1_b1  = tpr[np.searchsorted(fpr, 0.01)]
print(f'Tiempo: {t1-t0:.1f}s')
print(f'B1-BODMAS: AUC={auc_b1:.5f}  TPR@0.1%={tpr_01_b1:.4f}  TPR@1%={tpr_1_b1:.4f}')

print('\n=== COMPARATIVA FINAL COMPLETA ===')
print(f'B1 EMBER 2018 original:   AUC=0.99338  TPR@0.1%=0.8027  TPR@1%=0.9327')
print(f'B1 BODMAS temporal:       AUC={auc_b1:.5f}  TPR@0.1%={tpr_01_b1:.4f}  TPR@1%={tpr_1_b1:.4f}')
print(f'B2 PRISM limpio:          AUC=0.99885  TPR@0.1%=0.7210  TPR@1%=0.9843')
print()
print('NOTA: B2 PRISM sobre BODMAS pendiente de extraer matrices 2D')
print('      de los vectores 1D (requiere binarios originales).')
print('      El experimento justo PRISM vs EMBER sobre BODMAS')
print('      se completara cuando tengamos acceso a los binarios.')

m_b1.booster_.save_model('/home/prism/data/b1_bodmas_temporal.txt')
print('\nModelo guardado.')
