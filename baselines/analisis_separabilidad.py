import numpy as np
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

data_dir = '/home/prism/data/prism_matrices'

print('Cargando corpus PRISM...')
X_ben = np.load(data_dir + '/X_benign.npy')
X_mal = np.load(data_dir + '/X_malware.npy')
y_ben = np.load(data_dir + '/y_benign.npy')
y_mal = np.load(data_dir + '/y_malware.npy')

X = np.vstack([X_ben, X_mal])
y = np.concatenate([y_ben, y_mal])

# Deduplicar
hashes = np.array([hash(row.tobytes()) for row in
                   X.reshape(len(X), -1)])
_, idx = np.unique(hashes, return_index=True)
X = X[idx]
y = y[idx]

print(f'Muestras: {len(y)} | Shape matriz: {X.shape}')
print(f'N_max={X.shape[1]-1} secciones + 1 fila global | F={X.shape[2]} features')

N_MAX = X.shape[1] - 1
F     = X.shape[2]

FEATURE_NAMES = [
    'name0','name1','name2','name3','name4','name5','name6','name7',
    'raw_size','virt_size','ratio',
    'MEM_READ','MEM_WRITE','MEM_EXEC','MEM_DISC','CNT_CODE','CNT_DATA',
    'entropy','Q1','Q2','Q3','Q4',
    'position','unusual_name','WX_flag','zero_raw'
]

# ── 1. FISHER DISCRIMINANT RATIO por feature individual ──────────────────
print('\n=== 1. Fisher Discriminant Ratio - Features individuales ===')

fdr_results = []
for sec in range(N_MAX + 1):
    for f in range(F):
        vals = X[:, sec, f]
        v0 = vals[y == 0]
        v1 = vals[y == 1]
        mu0, mu1 = v0.mean(), v1.mean()
        s0,  s1  = v0.var(),  v1.var()
        denom = s0 + s1
        fdr = (mu1 - mu0)**2 / denom if denom > 1e-10 else 0.0
        label = 'GLOBAL' if sec == N_MAX else f'SEC{sec}'
        fdr_results.append((fdr, label, FEATURE_NAMES[f]))

fdr_results.sort(reverse=True)
print('Top 15 features por FDR:')
print(f'{"FDR":>8}  {"Seccion":<8}  {"Feature"}')
print('-' * 40)
for fdr, sec, feat in fdr_results[:15]:
    print(f'{fdr:8.4f}  {sec:<8}  {feat}')

# ── 2. MUTUAL INFORMATION por feature individual ─────────────────────────
print('\n=== 2. Mutual Information - Features individuales ===')

# Aplanar para MI
X_flat = X.reshape(len(X), -1)
print('Calculando MI (puede tardar 1-2 minutos)...')
mi_scores = mutual_info_classif(X_flat, y, random_state=42, n_neighbors=5)

mi_results = []
for i, mi in enumerate(mi_scores):
    sec = i // F
    f   = i  % F
    label = 'GLOBAL' if sec == N_MAX else f'SEC{sec}'
    mi_results.append((mi, label, FEATURE_NAMES[f]))

mi_results.sort(reverse=True)
print('Top 15 features por MI:')
print(f'{"MI":>8}  {"Seccion":<8}  {"Feature"}')
print('-' * 40)
for mi, sec, feat in mi_results[:15]:
    print(f'{mi:8.4f}  {sec:<8}  {feat}')

# ── 3. INFORMATION GAIN INTER-SECCION (Delta-I) ──────────────────────────
print('\n=== 3. Delta-I: Gain inter-seccion (top pares) ===')
print('Calculando MI individual por feature...')

mi_por_feature = {}
for sec in range(N_MAX + 1):
    for f in range(F):
        idx_flat = sec * F + f
        mi_por_feature[(sec, f)] = mi_scores[idx_flat]

print('Calculando Delta-I para pares inter-seccion...')
print('(comparando pares de secciones adyacentes)')

delta_i_results = []
# Solo secciones adyacentes para empezar (i, i+1)
for i in range(min(N_MAX - 1, 8)):
    for fa in range(F):
        for fb in range(F):
            # Feature conjunta: diferencia entre secciones
            diff = X[:, i, fa] - X[:, i+1, fb]
            diff = diff.reshape(-1, 1)
            mi_joint = mutual_info_classif(
                diff, y, random_state=42, n_neighbors=5)[0]
            mi_best = max(mi_por_feature[(i, fa)],
                         mi_por_feature[(i+1, fb)])
            delta_i = mi_joint - mi_best
            if delta_i > 0.01:
                delta_i_results.append((
                    delta_i, i, i+1,
                    FEATURE_NAMES[fa], FEATURE_NAMES[fb]
                ))

delta_i_results.sort(reverse=True)
print(f'\nPares con Delta-I > 0.01: {len(delta_i_results)}')
print(f'{"Delta-I":>8}  {"Sec_i":>6}  {"Sec_j":>6}  {"Feature_i":<16}  {"Feature_j"}')
print('-' * 60)
for di, si, sj, fa, fb in delta_i_results[:20]:
    print(f'{di:8.4f}  {si:>6}  {sj:>6}  {fa:<16}  {fb}')

# ── 4. RESUMEN ────────────────────────────────────────────────────────────
print('\n=== 4. RESUMEN PARA EL PAPER ===')
mean_fdr_sec    = np.mean([r[0] for r in fdr_results
                           if not r[1].startswith('G')])
mean_fdr_global = np.mean([r[0] for r in fdr_results
                           if r[1].startswith('G')])
mean_mi_sec     = np.mean([r[0] for r in mi_results
                           if not r[1].startswith('G')])
mean_mi_global  = np.mean([r[0] for r in mi_results
                           if r[1].startswith('G')])

print(f'FDR medio features de seccion:  {mean_fdr_sec:.4f}')
print(f'FDR medio features globales:    {mean_fdr_global:.4f}')
print(f'Ratio FDR sec/global:           {mean_fdr_sec/mean_fdr_global:.2f}x')
print()
print(f'MI medio features de seccion:   {mean_mi_sec:.4f}')
print(f'MI medio features globales:     {mean_mi_global:.4f}')
print(f'Ratio MI sec/global:            {mean_mi_sec/mean_mi_global:.2f}x')
print()
if delta_i_results:
    mean_delta = np.mean([r[0] for r in delta_i_results])
    max_delta  = delta_i_results[0][0]
    print(f'Delta-I maximo (mejor par):     {max_delta:.4f} bits')
    print(f'Delta-I medio (pares>0.01):     {mean_delta:.4f} bits')
    print(f'Pares con ganancia positiva:    {len(delta_i_results)}')
