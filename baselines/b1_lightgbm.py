import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import os, time

data_dir = '/home/prism/data/ember2018/ember2018/'

print('Cargando datos...')
X_train = np.memmap(data_dir+'X_train.dat', dtype=np.float32, mode='r').reshape(-1,2381)
y_train = np.memmap(data_dir+'y_train.dat', dtype=np.float32, mode='r')
X_test  = np.memmap(data_dir+'X_test.dat',  dtype=np.float32, mode='r').reshape(-1,2381)
y_test  = np.memmap(data_dir+'y_test.dat',  dtype=np.float32, mode='r')

# Solo muestras etiquetadas para train
mask = y_train != -1
X_tr = X_train[mask]
y_tr = y_train[mask]
print(f'Train etiquetado: {X_tr.shape[0]:,} muestras')

print('Entrenando LightGBM B1 (EMBER 1D)...')
t0 = time.time()
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    n_jobs=8,
    random_state=42,
    scale_pos_weight=(y_tr==0).sum()/(y_tr==1).sum()
)
model.fit(X_tr, y_tr)
t1 = time.time()
print(f'Tiempo de entrenamiento: {t1-t0:.1f} segundos')

print('Evaluando...')
y_prob = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_prob)
print(f'AUC-ROC: {auc:.5f}')

# TPR a FPR=0.1% y FPR=1%
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
idx_01 = np.searchsorted(fpr, 0.001)
idx_1  = np.searchsorted(fpr, 0.01)
print(f'TPR @ FPR=0.1%: {tpr[idx_01]:.4f}')
print(f'TPR @ FPR=1.0%: {tpr[idx_1]:.4f}')

model.booster_.save_model('/home/prism/data/b1_model.txt')
print('Modelo guardado en /home/prism/data/b1_model.txt')
