import os, gc, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil, joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import xgboost as xgb

warnings.filterwarnings('ignore')

# =============================================================================
ROAD_OUT  = '/Users/tarun/Desktop/road_output'
OTIDS_OUT = '/Users/tarun/Desktop/otids_output'
RES_DIR   = '/Users/tarun/Desktop/cross_vehicle_results'
# =============================================================================

os.makedirs(RES_DIR, exist_ok=True)

ROAD_FEAT_COLS = [
    'sig_mean','sig_std','sig_min','sig_max','sig_range','sig_entropy',
    'iat','iat_zscore','freq_dev',
    'hamming_dist','payload_change',
    'roll_iat_mean','roll_iat_std','roll_entropy_mean','roll_hamming_mean',
    'arb_id_enc','sig_0','sig_1','sig_2','sig_3'
]

OTIDS_FEAT_COLS = [
    'byte_0','byte_1','byte_2','byte_3','byte_4','byte_5','byte_6','byte_7',
    'byte_mean','byte_std','byte_min','byte_max','byte_range','byte_entropy',
    'iat','iat_zscore','freq_dev','hamming','dlc_change',
    'roll_iat_mean','roll_iat_std','roll_entropy_mean','arb_id_enc'
]

SHARED_COLS = ['iat','iat_zscore','freq_dev',
               'roll_iat_mean','roll_iat_std','roll_entropy_mean',
               'hamming_dist','arb_id_enc']

def ram():
    p = psutil.virtual_memory()
    print(f'  RAM  used={(p.total-p.available)/1e9:.1f}GB  avail={p.available/1e9:.1f}GB')

def evaluate(name, y_true, y_pred, y_proba):
    m = {
        'Model'    : name,
        'Accuracy' : round(accuracy_score(y_true, y_pred)*100, 2),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0)*100, 2),
        'Recall'   : round(recall_score(y_true, y_pred, zero_division=0)*100, 2),
        'F1'       : round(f1_score(y_true, y_pred, zero_division=0)*100, 2),
        'ROC-AUC'  : round(roc_auc_score(y_true, y_proba)*100
                           if len(np.unique(y_true)) > 1 else 0.0, 2)
    }
    print(f'    Acc={m["Accuracy"]:.1f}%  P={m["Precision"]:.1f}%  '
          f'R={m["Recall"]:.1f}%  F1={m["F1"]:.1f}%  AUC={m["ROC-AUC"]:.1f}%')
    return m

def balance(df, feat_cols, ratio=3):
    """Undersample majority class to ratio:1 to fix class imbalance."""
    atk  = df[df['label']==1]
    norm = df[df['label']==0]
    if len(atk) == 0:
        raise RuntimeError(
            'No attack samples in ROAD features! '
            'Delete /Users/tarun/Desktop/road_output/ and rerun nb1_road.py')
    n_norm = min(len(norm), len(atk) * ratio)
    return pd.concat([norm.sample(n_norm, random_state=42), atk]
                     ).sample(frac=1, random_state=42).reset_index(drop=True)

# =============================================================================
#  STEP 1 — Check required files
# =============================================================================
print('='*68)
print('  STEP 1 — Checking required files')
print('='*68)
for path, label in [
    (os.path.join(ROAD_OUT,  'road_features.parquet'),  'ROAD features'),
    (os.path.join(OTIDS_OUT, 'otids_features.parquet'), 'OTIDS features'),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing: {path}\nRun nb1_road.py / nb2_otids.py first.')
    sz = os.path.getsize(path)/1e6
    print(f'  OK  {label}  ({sz:.1f} MB)')

# =============================================================================
#  STEP 2 — Load & balance ROAD data
# =============================================================================
print('\n' + '='*68)
print('  STEP 2 — Loading ROAD features')
print('='*68)

road_raw = pd.read_parquet(os.path.join(ROAD_OUT, 'road_features.parquet'))
print(f'Raw ROAD: {road_raw.shape}  | attack={road_raw["label"].mean()*100:.2f}%')
print('Attack type breakdown:')
print(road_raw.groupby('attack_type')['label'].agg(['count','sum','mean']).round(3).to_string())

# Fill any missing columns
for col in ROAD_FEAT_COLS:
    if col not in road_raw.columns:
        road_raw[col] = 0.0

road_raw = road_raw[ROAD_FEAT_COLS + ['label','attack_type']].dropna()
road_raw[ROAD_FEAT_COLS] = road_raw[ROAD_FEAT_COLS].fillna(0).replace([np.inf,-np.inf], 0)

# Balance classes — keep all attacks, undersample normal 3:1
road_bal = balance(road_raw, ROAD_FEAT_COLS, ratio=3)
print(f'\nBalanced ROAD: {len(road_bal):,} | attack={road_bal["label"].mean()*100:.2f}%')
ram()

# =============================================================================
#  STEP 3 — Load & balance OTIDS data
# =============================================================================
print('\n' + '='*68)
print('  STEP 3 — Loading OTIDS features')
print('='*68)

otids_raw = pd.read_parquet(os.path.join(OTIDS_OUT, 'otids_features.parquet'))
print(f'Raw OTIDS: {otids_raw.shape} | attack={otids_raw["label"].mean()*100:.2f}%')

for col in OTIDS_FEAT_COLS:
    if col not in otids_raw.columns:
        otids_raw[col] = 0.0

otids_raw = otids_raw[OTIDS_FEAT_COLS + ['label','attack_type']].dropna()
otids_raw[OTIDS_FEAT_COLS] = otids_raw[OTIDS_FEAT_COLS].fillna(0).replace([np.inf,-np.inf], 0)
otids_bal = balance(otids_raw, OTIDS_FEAT_COLS, ratio=3)
print(f'Balanced OTIDS: {len(otids_bal):,} | attack={otids_bal["label"].mean()*100:.2f}%')

# Rename hamming for shared feature alignment
otids_bal['hamming_dist'] = otids_bal['hamming']
otids_raw['hamming_dist'] = otids_raw['hamming']
ram()

# =============================================================================
#  PART 1 — Train directly on ROAD 
# =============================================================================
print('\n' + '='*68)
print('  PART 1 — Direct Training on ROAD Dataset (all features)')
print('='*68)

X_road = road_bal[ROAD_FEAT_COLS].values.astype(np.float32)
y_road = road_bal['label'].values.astype(np.int32)
g_road = road_bal['attack_type'].values

X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
    X_road, y_road, g_road, test_size=0.3, random_state=42, stratify=y_road)

sc_road    = RobustScaler()
X_tr_sc    = sc_road.fit_transform(X_tr)
X_te_sc    = sc_road.transform(X_te)
joblib.dump(sc_road, os.path.join(RES_DIR, 'scaler_road.pkl'))
print(f'Train: {len(X_tr):,}  Test: {len(X_te):,}')
print(f'Train attack rate: {y_tr.mean()*100:.2f}%  Test: {y_te.mean()*100:.2f}%')

# Logistic Regression
print('\n--- Logistic Regression (ROAD direct) ---')
LR_PATH = os.path.join(RES_DIR, 'lr_road.pkl')
if os.path.exists(LR_PATH): lr = joblib.load(LR_PATH)
else:
    t0 = time.time()
    lr = LogisticRegression(C=1.0, max_iter=500, class_weight='balanced',
                             random_state=42, n_jobs=-1)
    lr.fit(X_tr_sc, y_tr); joblib.dump(lr, LR_PATH)
    print(f'  Trained in {time.time()-t0:.1f}s')
lr_pred  = lr.predict(X_te_sc)
lr_proba = lr.predict_proba(X_te_sc)[:,1]
m_lr = evaluate('LR [ROAD Direct]', y_te, lr_pred, lr_proba)

# Random Forest
print('\n--- Random Forest (ROAD direct) ---')
RF_PATH = os.path.join(RES_DIR, 'rf_road.pkl')
if os.path.exists(RF_PATH): rf = joblib.load(RF_PATH)
else:
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=3,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr); joblib.dump(rf, RF_PATH)
    print(f'  Trained in {time.time()-t0:.1f}s')
rf_pred  = rf.predict(X_te)
rf_proba = rf.predict_proba(X_te)[:,1]
m_rf = evaluate('RF [ROAD Direct]', y_te, rf_pred, rf_proba)

# XGBoost
print('\n--- XGBoost (ROAD direct) ---')
XGB_PATH = os.path.join(RES_DIR, 'xgb_road.pkl')
if os.path.exists(XGB_PATH): xgb_road = joblib.load(XGB_PATH)
else:
    t0 = time.time()
    spw = (y_tr==0).sum() / max((y_tr==1).sum(), 1)
    xgb_road = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb_road.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=50)
    joblib.dump(xgb_road, XGB_PATH)
    print(f'  Trained in {time.time()-t0:.1f}s')
xgb_pred  = xgb_road.predict(X_te)
xgb_proba = xgb_road.predict_proba(X_te)[:,1]
m_xgb = evaluate('XGBoost [ROAD Direct]', y_te, xgb_pred, xgb_proba)

# MLP
print('\n--- MLP (ROAD direct) ---')
MLP_PATH = os.path.join(RES_DIR, 'mlp_road.pkl')
if os.path.exists(MLP_PATH): mlp = joblib.load(MLP_PATH)
else:
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(128,64,32), activation='relu', solver='adam',
                         alpha=0.01, learning_rate_init=1e-3, max_iter=200,
                         early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=15, random_state=42)
    mlp.fit(X_tr_sc, y_tr); joblib.dump(mlp, MLP_PATH)
    print(f'  Trained in {time.time()-t0:.1f}s  (stopped at iter {mlp.n_iter_})')
mlp_pred  = mlp.predict(X_te_sc)
mlp_proba = mlp.predict_proba(X_te_sc)[:,1]
m_mlp = evaluate('MLP [ROAD Direct]', y_te, mlp_pred, mlp_proba)

print('\n--- ROAD Direct Training Summary ---')
direct_df = pd.DataFrame([m_lr, m_rf, m_xgb, m_mlp])
print(direct_df.to_string(index=False))
direct_df.to_csv(os.path.join(RES_DIR, 'road_direct_results.csv'), index=False)
ram()

# =============================================================================
#  PART 2 — Cross-Vehicle Transfer (OTIDS → ROAD, shared features only)
# =============================================================================
print('\n' + '='*68)
print('  PART 2 — Cross-Vehicle Transfer (OTIDS → ROAD)')
print('='*68)

# Verify shared cols exist in both
SHARED_COLS = [c for c in SHARED_COLS
               if c in otids_bal.columns and c in road_bal.columns]
print(f'Shared features ({len(SHARED_COLS)}): {SHARED_COLS}')

# Train OTIDS source model on shared features
X_otids_sh = otids_bal[SHARED_COLS].values.astype(np.float32)
y_otids_sh = otids_bal['label'].values.astype(np.int32)

sc_shared     = RobustScaler()
X_otids_sh_sc = sc_shared.fit_transform(X_otids_sh)
joblib.dump(sc_shared, os.path.join(RES_DIR, 'scaler_shared.pkl'))

SRC_XGB = os.path.join(RES_DIR, 'xgb_source.pkl')
if os.path.exists(SRC_XGB):
    print('Loading source XGBoost...'); xgb_src = joblib.load(SRC_XGB)
else:
    print(f'Training source XGBoost on OTIDS ({len(X_otids_sh):,} rows)...')
    t0 = time.time()
    spw = (y_otids_sh==0).sum() / max((y_otids_sh==1).sum(), 1)
    xgb_src = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb_src.fit(X_otids_sh_sc, y_otids_sh)
    joblib.dump(xgb_src, SRC_XGB); print(f'  Done in {time.time()-t0:.1f}s')

SRC_MLP = os.path.join(RES_DIR, 'mlp_source.pkl')
if os.path.exists(SRC_MLP):
    print('Loading source MLP...'); mlp_src = joblib.load(SRC_MLP)
else:
    print('Training source MLP on OTIDS...')
    t0 = time.time()
    mlp_src = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam',
                             alpha=0.005, learning_rate_init=1e-3, max_iter=200,
                             early_stopping=True, validation_fraction=0.1,
                             n_iter_no_change=15, random_state=42)
    mlp_src.fit(X_otids_sh_sc, y_otids_sh)
    joblib.dump(mlp_src, SRC_MLP); print(f'  Done in {time.time()-t0:.1f}s')

# Prepare ROAD shared features
X_road_sh    = road_bal[SHARED_COLS].values.astype(np.float32)
y_road_sh    = road_bal['label'].values.astype(np.int32)
g_road_sh    = road_bal['attack_type'].values
X_road_sh_sc = sc_shared.transform(X_road_sh)

# --- Experiment A: Zero-Shot ---
print('\n--- A: Zero-Shot (OTIDS model → ROAD, no ROAD data) ---')
zs_xgb_pred  = xgb_src.predict(X_road_sh_sc)
zs_xgb_proba = xgb_src.predict_proba(X_road_sh_sc)[:,1]
zs_mlp_pred  = mlp_src.predict(X_road_sh_sc)
zs_mlp_proba = mlp_src.predict_proba(X_road_sh_sc)[:,1]
m_zs_xgb = evaluate('XGBoost [Zero-Shot]', y_road_sh, zs_xgb_pred, zs_xgb_proba)
m_zs_mlp = evaluate('MLP [Zero-Shot]',     y_road_sh, zs_mlp_pred, zs_mlp_proba)

# --- Experiment B: Few-Shot (OTIDS + 10% ROAD) ---
print('\n--- B: Few-Shot (OTIDS + 10% ROAD) ---')
road_10pct, road_90pct = train_test_split(
    road_bal, test_size=0.9, random_state=42, stratify=road_bal['label'])

# Combine OTIDS and 10% ROAD (both scaled with shared scaler)
X_road10_sc = sc_shared.transform(road_10pct[SHARED_COLS].values.astype(np.float32))
y_road10    = road_10pct['label'].values.astype(np.int32)
X_comb10    = np.vstack([X_otids_sh_sc, X_road10_sc])
y_comb10    = np.concatenate([y_otids_sh, y_road10])
idx = np.random.RandomState(42).permutation(len(X_comb10))
X_comb10, y_comb10 = X_comb10[idx], y_comb10[idx]

X_test90 = sc_shared.transform(road_90pct[SHARED_COLS].values.astype(np.float32))
y_test90 = road_90pct['label'].values.astype(np.int32)
del road_10pct, road_90pct; gc.collect()

FT10_XGB = os.path.join(RES_DIR, 'xgb_ft10.pkl')
if os.path.exists(FT10_XGB): xgb_ft10 = joblib.load(FT10_XGB)
else:
    spw10 = (y_comb10==0).sum() / max((y_comb10==1).sum(), 1)
    xgb_ft10 = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   scale_pos_weight=spw10, eval_metric='logloss',
                                   random_state=42, n_jobs=-1)
    xgb_ft10.fit(X_comb10, y_comb10); joblib.dump(xgb_ft10, FT10_XGB)

FT10_MLP = os.path.join(RES_DIR, 'mlp_ft10.pkl')
if os.path.exists(FT10_MLP): mlp_ft10 = joblib.load(FT10_MLP)
else:
    mlp_ft10 = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam',
                               alpha=0.005, max_iter=200, early_stopping=True,
                               validation_fraction=0.1, n_iter_no_change=15, random_state=42)
    mlp_ft10.fit(X_comb10, y_comb10); joblib.dump(mlp_ft10, FT10_MLP)

ft10_xgb_proba = xgb_ft10.predict_proba(X_test90)[:,1]
ft10_mlp_proba = mlp_ft10.predict_proba(X_test90)[:,1]
m_ft10_xgb = evaluate('XGBoost [Few-Shot 10%]', y_test90, xgb_ft10.predict(X_test90), ft10_xgb_proba)
m_ft10_mlp = evaluate('MLP [Few-Shot 10%]',     y_test90, mlp_ft10.predict(X_test90), ft10_mlp_proba)

# --- Experiment C: Full Transfer (OTIDS + 50% ROAD) ---
print('\n--- C: Full Transfer (OTIDS + 50% ROAD) ---')
road_50a, road_50b = train_test_split(
    road_bal, test_size=0.5, random_state=42, stratify=road_bal['label'])

X_road50_sc = sc_shared.transform(road_50a[SHARED_COLS].values.astype(np.float32))
y_road50    = road_50a['label'].values.astype(np.int32)
X_comb50    = np.vstack([X_otids_sh_sc, X_road50_sc])
y_comb50    = np.concatenate([y_otids_sh, y_road50])
idx50 = np.random.RandomState(42).permutation(len(X_comb50))
X_comb50, y_comb50 = X_comb50[idx50], y_comb50[idx50]

X_test50 = sc_shared.transform(road_50b[SHARED_COLS].values.astype(np.float32))
y_test50 = road_50b['label'].values.astype(np.int32)
del road_50a, road_50b; gc.collect()

FT50_XGB = os.path.join(RES_DIR, 'xgb_ft50.pkl')
if os.path.exists(FT50_XGB): xgb_ft50 = joblib.load(FT50_XGB)
else:
    spw50 = (y_comb50==0).sum() / max((y_comb50==1).sum(), 1)
    xgb_ft50 = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   scale_pos_weight=spw50, eval_metric='logloss',
                                   random_state=42, n_jobs=-1)
    xgb_ft50.fit(X_comb50, y_comb50); joblib.dump(xgb_ft50, FT50_XGB)

FT50_MLP = os.path.join(RES_DIR, 'mlp_ft50.pkl')
if os.path.exists(FT50_MLP): mlp_ft50 = joblib.load(FT50_MLP)
else:
    mlp_ft50 = MLPClassifier(hidden_layer_sizes=(128,64,32), activation='relu', solver='adam',
                               alpha=0.005, max_iter=200, early_stopping=True,
                               validation_fraction=0.1, n_iter_no_change=15, random_state=42)
    mlp_ft50.fit(X_comb50, y_comb50); joblib.dump(mlp_ft50, FT50_MLP)

ft50_xgb_proba = xgb_ft50.predict_proba(X_test50)[:,1]
ft50_mlp_proba = mlp_ft50.predict_proba(X_test50)[:,1]
m_ft50_xgb = evaluate('XGBoost [Full-Transfer 50%]', y_test50, xgb_ft50.predict(X_test50), ft50_xgb_proba)
m_ft50_mlp = evaluate('MLP [Full-Transfer 50%]',     y_test50, mlp_ft50.predict(X_test50), ft50_mlp_proba)
ram()

# =============================================================================
#  PART 3 — Masquerade Detection
# =============================================================================
print('\n' + '='*68)
print('  PART 3 — Masquerade Attack Detection')
print('='*68)

MASQ = ['Speedometer_Masquerade','Engine_Masquerade','Correlated_Spoofing',
        'ReverseLight_Off','ReverseLight_On','Unknown_Attack','Accelerator_Attack']

atk_masq  = road_raw[road_raw['attack_type'].isin(MASQ) & (road_raw['label']==1)]
norm_masq = road_raw[road_raw['label']==0]
print(f'Masquerade attacks: {len(atk_masq):,}')
print(f'Types: {atk_masq["attack_type"].value_counts().to_string()}')

if len(atk_masq) == 0:
    print('No masquerade labels — using all attack rows')
    atk_masq = road_raw[road_raw['label']==1]

n_norm   = min(len(norm_masq), len(atk_masq)*3)
masq_df  = pd.concat([norm_masq.sample(n_norm, random_state=42), atk_masq]
                     ).sample(frac=1, random_state=42).reset_index(drop=True)
for col in ROAD_FEAT_COLS:
    if col not in masq_df.columns: masq_df[col] = 0.0
masq_df[ROAD_FEAT_COLS] = masq_df[ROAD_FEAT_COLS].fillna(0).replace([np.inf,-np.inf], 0)
del norm_masq, atk_masq; gc.collect()

print(f'Masquerade eval set: {len(masq_df):,} | attack={masq_df["label"].mean()*100:.1f}%')

X_masq    = masq_df[ROAD_FEAT_COLS].values.astype(np.float32)
y_masq    = masq_df['label'].values.astype(np.int32)
X_masq_sc = sc_road.transform(X_masq)

X_masq_sh    = masq_df[SHARED_COLS].values.astype(np.float32)
X_masq_sh_sc = sc_shared.transform(X_masq_sh)

# XGBoost directly trained on ROAD (best model)
masq_xgb_pred  = xgb_road.predict(X_masq)
masq_xgb_proba = xgb_road.predict_proba(X_masq)[:,1]
m_masq_xgb_direct = evaluate('XGBoost [Masquerade, ROAD-Direct]', y_masq, masq_xgb_pred, masq_xgb_proba)

# MLP directly on ROAD
masq_mlp_pred  = mlp.predict(X_masq_sc)
masq_mlp_proba = mlp.predict_proba(X_masq_sc)[:,1]
m_masq_mlp_direct = evaluate('MLP [Masquerade, ROAD-Direct]', y_masq, masq_mlp_pred, masq_mlp_proba)

# Zero-shot (OTIDS model)
masq_zs_pred  = xgb_src.predict(X_masq_sh_sc)
masq_zs_proba = xgb_src.predict_proba(X_masq_sh_sc)[:,1]
m_masq_zs = evaluate('XGBoost [Masquerade, Zero-Shot]', y_masq, masq_zs_pred, masq_zs_proba)

# Full-transfer
masq_ft_pred  = xgb_ft50.predict(X_masq_sh_sc)
masq_ft_proba = xgb_ft50.predict_proba(X_masq_sh_sc)[:,1]
m_masq_ft = evaluate('XGBoost [Masquerade, Full-Transfer]', y_masq, masq_ft_pred, masq_ft_proba)

print('\n--- Masquerade Detection Summary ---')
print(pd.DataFrame([m_masq_xgb_direct, m_masq_mlp_direct, m_masq_zs, m_masq_ft]).to_string(index=False))

# =============================================================================
#  PART 4 — Full Results Table
# =============================================================================
print('\n' + '='*78)
print('  FULL RESULTS — NB4 CROSS-VEHICLE GENERALIZATION')
print('='*78)

all_results = [
    m_lr, m_rf, m_xgb, m_mlp,                           # Part 1: ROAD direct
    m_zs_xgb, m_zs_mlp,                                  # Part 2: Zero-shot
    m_ft10_xgb, m_ft10_mlp,                              # Part 2: Few-shot
    m_ft50_xgb, m_ft50_mlp,                              # Part 2: Full transfer
    m_masq_xgb_direct, m_masq_mlp_direct,               # Part 3: Masquerade direct
    m_masq_zs, m_masq_ft,                                # Part 3: Masquerade transfer
]
res_df = pd.DataFrame(all_results)
res_df.to_csv(os.path.join(RES_DIR,'cross_vehicle_results.csv'), index=False)
print(res_df.to_string(index=False))
print('='*78)

# =============================================================================
#  PLOTS
# =============================================================================
print('\nGenerating plots...')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('NB4 — Cross-Vehicle Generalization & Masquerade Detection',
             fontsize=14, fontweight='bold')

# Plot 1: ROAD Direct training — all 4 models
metrics = ['Accuracy','Precision','Recall','F1','ROC-AUC']
models  = [m_lr, m_rf, m_xgb, m_mlp]
names   = ['LR','RF','XGBoost','MLP']
colors  = ['#3498db','#2ecc71','#e74c3c','#9b59b6']
x = np.arange(len(metrics)); w = 0.18
for idx, (m, n, c) in enumerate(zip(models, names, colors)):
    vals = [m[k] for k in metrics]
    axes[0,0].bar(x+idx*w, vals, w, label=n, color=c, alpha=0.85)
axes[0,0].set_xticks(x+w*1.5); axes[0,0].set_xticklabels(metrics, fontsize=9)
axes[0,0].set_ylim(0, 112); axes[0,0].set_ylabel('Score (%)')
axes[0,0].set_title('Part 1: ROAD Direct Training (All Features)')
axes[0,0].legend(fontsize=8)
axes[0,0].axhline(75, color='gray', ls='--', lw=1, label='75% min')
axes[0,0].axhline(90, color='gray', ls=':', lw=1)
axes[0,0].grid(axis='y', alpha=0.3)

# Plot 2: Cross-vehicle transfer F1 progression
stages   = ['Zero-Shot\n(0%)', 'Few-Shot\n(10%)', 'Full-Transfer\n(50%)']
f1_xgb_t = [m_zs_xgb['F1'], m_ft10_xgb['F1'], m_ft50_xgb['F1']]
f1_mlp_t = [m_zs_mlp['F1'], m_ft10_mlp['F1'], m_ft50_mlp['F1']]
x2 = np.arange(len(stages)); w2 = 0.35
axes[0,1].bar(x2-w2/2, f1_xgb_t, w2, label='XGBoost', color='#3498db', alpha=0.85)
axes[0,1].bar(x2+w2/2, f1_mlp_t, w2, label='MLP',     color='#e74c3c', alpha=0.85)
axes[0,1].set_xticks(x2); axes[0,1].set_xticklabels(stages, fontsize=9)
axes[0,1].set_ylim(0, 110); axes[0,1].set_ylabel('F1 Score (%)')
axes[0,1].set_title('Part 2: Cross-Vehicle Transfer F1 (Shared Features)')
axes[0,1].legend(fontsize=9); axes[0,1].grid(axis='y', alpha=0.3)
for i,(vx,vm) in enumerate(zip(f1_xgb_t, f1_mlp_t)):
    axes[0,1].text(i-w2/2, vx+1, f'{vx:.1f}%', ha='center', fontsize=8, fontweight='bold')
    axes[0,1].text(i+w2/2, vm+1, f'{vm:.1f}%', ha='center', fontsize=8, fontweight='bold')

# Plot 3: Accuracy comparison — Direct vs Transfer
direct_accs   = [m_xgb['Accuracy'], m_mlp['Accuracy']]
transfer_accs = [m_ft50_xgb['Accuracy'], m_ft50_mlp['Accuracy']]
zs_accs       = [m_zs_xgb['Accuracy'], m_zs_mlp['Accuracy']]
x3  = np.arange(2); labels3 = ['XGBoost','MLP']; w3 = 0.25
axes[1,0].bar(x3-w3,   zs_accs,       w3, label='Zero-Shot',      color='#e74c3c', alpha=0.85)
axes[1,0].bar(x3,      transfer_accs, w3, label='Full-Transfer',   color='#f39c12', alpha=0.85)
axes[1,0].bar(x3+w3,   direct_accs,   w3, label='ROAD Direct',     color='#2ecc71', alpha=0.85)
axes[1,0].set_xticks(x3); axes[1,0].set_xticklabels(labels3)
axes[1,0].set_ylim(0, 110); axes[1,0].set_ylabel('Accuracy (%)')
axes[1,0].set_title('Accuracy: Zero-Shot vs Transfer vs Direct')
axes[1,0].axhspan(75, 90, color='yellow', alpha=0.2, label='Target 75-90%')
axes[1,0].legend(fontsize=8); axes[1,0].grid(axis='y', alpha=0.3)
for x_pos, vals in [(x3-w3, zs_accs), (x3, transfer_accs), (x3+w3, direct_accs)]:
    for xp, v in zip(x_pos, vals):
        axes[1,0].text(xp, v+1, f'{v:.1f}%', ha='center', fontsize=8, fontweight='bold')

# Plot 4: Masquerade detection comparison
masq_names = ['XGBoost\nROAD-Direct', 'MLP\nROAD-Direct',
              'XGBoost\nZero-Shot',   'XGBoost\nFull-Transfer']
masq_f1s   = [m_masq_xgb_direct['F1'], m_masq_mlp_direct['F1'],
              m_masq_zs['F1'],          m_masq_ft['F1']]
masq_accs  = [m_masq_xgb_direct['Accuracy'], m_masq_mlp_direct['Accuracy'],
              m_masq_zs['Accuracy'],          m_masq_ft['Accuracy']]
masq_cols  = ['#2ecc71','#27ae60','#e74c3c','#f39c12']
x4 = np.arange(len(masq_names)); w4 = 0.35
axes[1,1].bar(x4-w4/2, masq_accs, w4, label='Accuracy', color='#3498db', alpha=0.85)
axes[1,1].bar(x4+w4/2, masq_f1s,  w4, label='F1',       color='#e74c3c', alpha=0.85)
axes[1,1].set_xticks(x4); axes[1,1].set_xticklabels(masq_names, fontsize=8)
axes[1,1].set_ylim(0, 110); axes[1,1].set_ylabel('Score (%)')
axes[1,1].set_title('Part 3: Masquerade Attack Detection')
axes[1,1].legend(fontsize=9); axes[1,1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RES_DIR,'cross_vehicle_results.png'), dpi=150, bbox_inches='tight')
plt.show(); print('Saved: cross_vehicle_results.png')

# Confusion matrices — ROAD direct best model (XGBoost)
fig, axes2 = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Confusion Matrices — ROAD Direct (XGBoost) & Masquerade', fontsize=13, fontweight='bold')
for ax, (name, yt, yp) in zip(axes2, [
    ('XGBoost ROAD Direct', y_te, xgb_pred),
    ('Masquerade Detection', y_masq, masq_xgb_pred),
]):
    cm  = confusion_matrix(yt, yp)
    pct = cm.astype(float)/cm.sum()*100
    ann = np.array([[f'{v}\n({p:.1f}%)' for v,p in zip(rc,rp)] for rc,rp in zip(cm,pct)])
    sns.heatmap(cm, annot=ann, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Normal','Attack'], yticklabels=['Normal','Attack'],
                annot_kws={'size':10})
    ax.set_title(f'{name}\nacc={accuracy_score(yt,yp)*100:.1f}%  F1={f1_score(yt,yp)*100:.1f}%',
                 fontsize=11)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR,'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.show(); print('Saved: confusion_matrices.png')

# =============================================================================
#  FINAL SUMMARY
# =============================================================================
print('\n' + '='*78)
print('  FINAL SUMMARY — NB4')
print('='*78)
print(f'\n  Part 1 — ROAD Direct Training (all {len(ROAD_FEAT_COLS)} features):')
print(f'  {"Model":<30s}  {"Acc":>7s}  {"F1":>7s}  {"AUC":>7s}')
print(f'  {"-"*30}  {"-"*7}  {"-"*7}  {"-"*7}')
for m in [m_lr, m_rf, m_xgb, m_mlp]:
    tag = '[TARGET]' if 75<=m['Accuracy']<=90 else ('[ABOVE]' if m['Accuracy']>90 else '[BELOW]')
    print(f'  {m["Model"]:<30s}  {m["Accuracy"]:>6.1f}%  {m["F1"]:>6.1f}%  {m["ROC-AUC"]:>6.1f}%  {tag}')

print(f'\n  Part 2 — Cross-Vehicle Transfer ({len(SHARED_COLS)} shared features):')
print(f'  {"Model":<35s}  {"Acc":>7s}  {"F1":>7s}  {"AUC":>7s}')
print(f'  {"-"*35}  {"-"*7}  {"-"*7}  {"-"*7}')
for m in [m_zs_xgb, m_zs_mlp, m_ft10_xgb, m_ft10_mlp, m_ft50_xgb, m_ft50_mlp]:
    print(f'  {m["Model"]:<35s}  {m["Accuracy"]:>6.1f}%  {m["F1"]:>6.1f}%  {m["ROC-AUC"]:>6.1f}%')

print(f'\n  Part 3 — Masquerade Detection:')
print(f'  {"Model":<42s}  {"Acc":>7s}  {"F1":>7s}  {"AUC":>7s}')
print(f'  {"-"*42}  {"-"*7}  {"-"*7}  {"-"*7}')
for m in [m_masq_xgb_direct, m_masq_mlp_direct, m_masq_zs, m_masq_ft]:
    print(f'  {m["Model"]:<42s}  {m["Accuracy"]:>6.1f}%  {m["F1"]:>6.1f}%  {m["ROC-AUC"]:>6.1f}%')

print(f'\n  Output: {RES_DIR}')
print('='*78)
ram()
print('\nNB4 complete.')