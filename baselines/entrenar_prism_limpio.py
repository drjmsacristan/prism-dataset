import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
import time

data_dir = '/home/prism/data/prism_matrices'

print('Cargando corpus PRISM...')
X_ben = np.load(data_dir + '/X_benign.npy')
X_mal = np.load(data_dir + '/X_malware.npy')
y_ben = np.load(data_dir + '/y_benign.npy')
y_mal = np.load(data_dir + '/y_malware.npy')

X_ben_flat = X_ben.reshape(X_ben.shape[0], -1)
X_mal_flat = X_mal.reshape(X_mal.shape[0], -1)

X = np.vstack([X_ben_flat, X_mal_flat])
y = np.concatenate([y_ben, y_mal])

# PASO 1: Eliminar duplicados exactos
print('\nEliminando duplicados...')
hashes = np.array([hash(row.tobytes()) for row in X])
_, idx_unicos = np.unique(hashes, return_index=True)
X = X[idx_unicos]
y = y[idx_unicos]
print(f'Muestras tras deduplicacion: {len(y):,}')
print(f'Benigno: {(y==0).sum():,} | Malware: {(y==1).sum():,}')

# PASO 2: Split sin solapamiento - separar indices por clase
np.random.seed(42)
idx_ben = np.where(y == 0)[0]
idx_mal = np.where(y == 1)[0]
np.random.shuffle(idx_ben)
np.random.shuffle(idx_mal)

# 80% train, 20% test por clase (stratified manual)
n_ben_test = int(len(idx_ben) * 0.2)
n_mal_test = int(len(idx_mal) * 0.2)

test_idx  = np.concatenate([idx_ben[:n_ben_test], idx_mal[:n_mal_test]])
train_idx = np.concatenate([idx_ben[n_ben_test:], idx_mal[n_mal_test:]])

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

# Verificar sin solapamiento
train_h = set(hash(r.tobytes()) for r in X_train)
test_h  = set(hash(r.tobytes()) for r in X_test)
print(f'Solapamiento tras limpieza: {len(train_h & test_h)}')

print(f'\nTrain: {len(y_train):,} | Malware: {y_train.sum():.0f} ({100*y_train.mean():.1f}%)')
print(f'Test:  {len(y_test):,}  | Malware: {y_test.sum():.0f}  ({100*y_test.mean():.1f}%)')

# PASO 3: Entrenar con class_weight correcto
print('\nEntrenando LightGBM B2 limpio...')
ratio = (y_train==0).sum() / (y_train==1).sum()
print(f'scale_pos_weight: {ratio:.2f}')

t0 = time.time()
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    n_jobs=8,
    random_state=42,
    scale_pos_weight=ratio
)
model.fit(X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(100)])
t1 = time.time()
print(f'Tiempo: {t1-t0:.1f}s | Arboles usados: {model.best_iteration_}')

# PASO 4: Evaluar
y_prob = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)
idx_01 = np.searchsorted(fpr, 0.001)
idx_1  = np.searchsorted(fpr, 0.01)

print(f'\n=== RESULTADOS LIMPIOS ===')
print(f'AUC-ROC:        {auc:.5f}')
print(f'TPR @ FPR=0.1%: {tpr[idx_01]:.4f}')
print(f'TPR @ FPR=1.0%: {tpr[idx_1]:.4f}')
print(f'\nComparativa:')
print(f'B1 EMBER 1D:    AUC=0.99338  TPR@0.1%=0.8027  TPR@1%=0.9327')
print(f'B2 PRISM limpio: AUC={auc:.5f}  TPR@0.1%={tpr[idx_01]:.4f}  TPR@1%={tpr[idx_1]:.4f}')

model.booster_.save_model('/home/prism/data/b2_prism_limpio.txt')
print('\nModelo guardado.')
