import os, gc, glob, re, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil, joblib
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
import xgboost as xgb

warnings.filterwarnings('ignore')
ROAD_DIR = 'data/road'
OUT_DIR  = 'outputs/road'

os.makedirs(OUT_DIR, exist_ok=True)

RAW_PATH   = os.path.join(OUT_DIR, 'road_raw.parquet')
CLEAN_PATH = os.path.join(OUT_DIR, 'road_cleaned.parquet')
FEAT_PATH  = os.path.join(OUT_DIR, 'road_features.parquet')

FEAT_COLS = [
    'sig_mean','sig_std','sig_min','sig_max','sig_range','sig_entropy',
    'iat','iat_zscore','freq_dev',
    'hamming_dist','payload_change',
    'roll_iat_mean','roll_iat_std','roll_entropy_mean','roll_hamming_mean',
    'arb_id_enc',
    'sig_0','sig_1','sig_2','sig_3'
]

# Attack type map from FILENAME
ATTACK_MAP = {
    'correlated_signal' : 'Correlated_Spoofing',
    'fuzzing'           : 'Fuzzing',
    'max_speedometer'   : 'Speedometer_Masquerade',
    'max_engine'        : 'Engine_Masquerade',
    'reverse_light_off' : 'ReverseLight_Off',
    'reverse_light_on'  : 'ReverseLight_On',
    'accelerator'       : 'Accelerator_Attack',
}

def ram():
    p = psutil.virtual_memory()
    print(f'  RAM  used={(p.total-p.available)/1e9:.1f}GB  avail={p.available/1e9:.1f}GB')

def get_attack_type(fpath):
    fname = os.path.basename(fpath).lower()
    for k, v in ATTACK_MAP.items():
        if k in fname:
            return v
    if 'attack' in fname or 'masquerade' in fname:
        return 'Unknown_Attack'
    return 'Normal'

# =============================================================================
#  STEP 1 — Find CSV files
# =============================================================================
print('='*68)
print('  STEP 1 — Finding CSV files')
print('='*68)

all_csvs = glob.glob(os.path.join(ROAD_DIR,'signal_extractions','**','*.csv'), recursive=True)
if not all_csvs:
    all_csvs = glob.glob(os.path.join(ROAD_DIR,'**','*.csv'), recursive=True)
all_csvs = [f for f in all_csvs if os.path.getsize(f) > 5_000 and 'data_table' not in f.lower()]

if not all_csvs:
    raise FileNotFoundError(f'No CSV files found under {ROAD_DIR}')

atk_csvs  = sorted([f for f in all_csvs if get_attack_type(f) != 'Normal'])
norm_csvs = sorted([f for f in all_csvs if get_attack_type(f) == 'Normal'])

print(f'Attack CSVs: {len(atk_csvs)}  |  Normal CSVs: {len(norm_csvs)}')
for f in atk_csvs:
    print(f'  ATK  {os.path.basename(f):60s} -> {get_attack_type(f)}')
for f in norm_csvs[:5]:
    print(f'  NRM  {os.path.basename(f):60s} -> Normal')

# Peek at format
print('\nPeeking at format:')
peek = pd.read_csv(all_csvs[0], nrows=3)
print(f'File: {os.path.basename(all_csvs[0])}')
print(f'Cols: {list(peek.columns)}')
print(peek.to_string())
ram()

# =============================================================================
#  STEP 2 — Parse CSVs
# =============================================================================
print('\n' + '='*68)
print('  STEP 2 — Parsing CSV files')
print('='*68)

CHUNK   = 100_000
MAX_PER = 500_000

def parse_road_csv(fpath, attack_type):
    chunks = []; rows_read = 0
    for chunk in pd.read_csv(fpath, chunksize=CHUNK, low_memory=False, on_bad_lines='skip'):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        cl = {c: c.lower().strip() for c in chunk.columns}

        # Time
        tc = next((c for c in chunk.columns if cl[c] in ('time','timestamp','t','time_s')), None)
        chunk['time'] = pd.to_numeric(
            chunk[tc] if tc else np.arange(rows_read, rows_read+len(chunk))*0.001,
            errors='coerce')

        # ── LABEL: use existing Label column if present ──────────────────────
        # ROAD signal_extractions have Label: 0=normal, 1=attack
        lc = next((c for c in chunk.columns if cl[c] in ('label','flag','class')), None)
        if lc is not None:
            chunk['label'] = pd.to_numeric(chunk[lc], errors='coerce').fillna(0).astype(int)
            # If label col exists but all zeros AND this is an attack file,
            # it means the label col is not populated — use attack_type heuristic
            if chunk['label'].sum() == 0 and attack_type != 'Normal':
                # Assign label=1 to the SECOND HALF of the file (attack onset)
                mid = len(chunk) // 2
                chunk['label'] = 0
                chunk.iloc[mid:, chunk.columns.get_loc('label')] = 1
        else:
            # No label column — infer from attack_type
            if attack_type == 'Normal':
                chunk['label'] = 0
            else:
                mid = len(chunk) // 2
                chunk['label'] = 0
                chunk.iloc[mid:, chunk.columns.get_loc('label')] = 1

        # Arb ID
        ic = next((c for c in chunk.columns
                   if cl[c] in ('id','arb_id','can_id','arbitration_id')), None)
        chunk['arb_id'] = chunk[ic].astype(str).str.strip().str.lower() if ic else 'unknown'

        # Signal columns
        scols = [c for c in chunk.columns if re.match(r'(?i)^signal[-_\s]\d', c)]
        if scols:
            sdata = chunk[scols].apply(pd.to_numeric, errors='coerce').fillna(0)
            for i in range(4):
                chunk[f'sig_{i}'] = sdata.iloc[:,i] if i < len(scols) else 0.0
        else:
            for i in range(4): chunk[f'sig_{i}'] = 0.0

        chunk['attack_type'] = attack_type
        keep = ['time','arb_id','label','attack_type'] + [f'sig_{i}' for i in range(4)]
        chunk.dropna(subset=['time'], inplace=True)
        chunks.append(chunk[keep].copy())
        rows_read += len(chunk)
        if rows_read >= MAX_PER: break

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

if os.path.exists(RAW_PATH):
    print(f'Already exists — skipping: {RAW_PATH}')
else:
    all_frames = []

    print('\nLoading attack CSVs...')
    for f in atk_csvs:
        atype = get_attack_type(f)
        try:
            df = parse_road_csv(f, atype)
            if len(df):
                all_frames.append(df)
                print(f'  OK  {os.path.basename(f):55s} {len(df):>7,} rows | '
                      f'attack={df["label"].sum():,} ({df["label"].mean()*100:.1f}%)  [{atype}]')
        except Exception as e:
            print(f'  ERR {os.path.basename(f)}: {e}')

    print('\nLoading normal CSVs...')
    for f in norm_csvs:
        try:
            df = parse_road_csv(f, 'Normal')
            if len(df):
                all_frames.append(df)
                print(f'  OK  {os.path.basename(f):55s} {len(df):>7,} rows | Normal')
        except Exception as e:
            print(f'  ERR {os.path.basename(f)}: {e}')

    if not all_frames:
        raise RuntimeError('No data loaded! Check CSV format.')

    raw = pd.concat(all_frames, ignore_index=True)
    raw.to_parquet(RAW_PATH, index=False)
    print(f'\nSaved raw: {len(raw):,} rows | attack={raw["label"].sum():,} ({raw["label"].mean()*100:.2f}%)')
    print('By attack type:')
    print(raw.groupby('attack_type')['label'].agg(['count','sum','mean']).round(3).to_string())
    del all_frames, raw; gc.collect()
ram()

# =============================================================================
#  STEP 3 — Clean
# =============================================================================
print('\n' + '='*68)
print('  STEP 3 — Cleaning')
print('='*68)

if os.path.exists(CLEAN_PATH):
    print(f'Already exists — skipping: {CLEAN_PATH}')
else:
    df = pd.read_parquet(RAW_PATH)
    print(f'Raw: {df.shape} | attack={df["label"].mean()*100:.2f}%')

    b = len(df); df.dropna(subset=['time'], inplace=True)
    print(f'[1] Drop null time    : -{b-len(df):,}')
    b = len(df); df = df[df['arb_id'].notna() & (df['arb_id'].str.strip() != '')]
    print(f'[2] Drop empty arb_id : -{b-len(df):,}')
    b = len(df)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df[df['label'].isin([0,1])]; df['label'] = df['label'].astype(np.int8)
    print(f'[3] Binary label      : -{b-len(df):,}')
    b = len(df)
    df.drop_duplicates(subset=['time','arb_id','label'], keep='first', inplace=True)
    print(f'[4] Dedup             : -{b-len(df):,}')
    for col in [f'sig_{i}' for i in range(4)]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        m, s = df[col].mean(), df[col].std()
        if s > 0: df[col] = df[col].clip(m-4*s, m+4*s)

    df['arb_id'] = df['arb_id'].astype(str).str.strip().str.lower()
    df.sort_values('time', inplace=True); df.reset_index(drop=True, inplace=True)

    print(f'\nFinal: {len(df):,} rows | attack={df["label"].sum():,} ({df["label"].mean()*100:.2f}%)')
    print(df.groupby('attack_type')['label'].agg(['count','sum','mean']).round(3).to_string())
    df.to_parquet(CLEAN_PATH, index=False)
    print(f'Saved: {CLEAN_PATH}')
    del df; gc.collect()
ram()

# =============================================================================
#  STEP 4 — Feature Engineering
# =============================================================================
print('\n' + '='*68)
print('  STEP 4 — Feature Engineering')
print('='*68)

def signal_entropy(vals):
    a = np.array(vals, dtype=np.float32)
    if a.sum() == 0: return 0.0
    _, c = np.unique(np.round(a, 1), return_counts=True)
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))

def engineer_features(df):
    df   = df.copy().sort_values('time').reset_index(drop=True)
    n    = len(df)
    smat = df[[f'sig_{i}' for i in range(4)]].values.astype(np.float32)

    print('  [A] Signal statistics...')
    df['sig_mean']    = smat.mean(axis=1)
    df['sig_std']     = smat.std(axis=1)
    df['sig_min']     = smat.min(axis=1)
    df['sig_max']     = smat.max(axis=1)
    df['sig_range']   = df['sig_max'] - df['sig_min']
    df['sig_entropy'] = [signal_entropy(smat[i]) for i in range(n)]

    print('  [B] Per-ID temporal features...')
    iat   = np.zeros(n, np.float32)
    hdist = np.zeros(n, np.float32)
    pchg  = np.zeros(n, np.float32)
    lt, ls = {}, {}
    for i in range(n):
        cid = df.at[i,'arb_id']; ts = df.at[i,'time']; sv = smat[i]
        if cid in lt:
            iat[i]   = ts - lt[cid]
            hdist[i] = float(np.sum(np.abs(np.round(sv) - np.round(ls[cid]))))
            pchg[i]  = float(not np.allclose(sv, ls[cid], atol=0.5))
        lt[cid] = ts; ls[cid] = sv
    df['iat']            = iat
    df['hamming_dist']   = hdist
    df['payload_change'] = pchg
    df['iat_zscore'] = df.groupby('arb_id')['iat'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9))
    df['iat_inv']  = 1.0 / df['iat'].clip(lower=1e-6)
    df['freq_dev'] = df.groupby('arb_id')['iat_inv'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9))
    df.drop(columns=['iat_inv'], inplace=True)

    print('  [C] Rolling window...')
    df['roll_iat_mean']     = df['iat'].rolling(20, min_periods=1).mean()
    df['roll_iat_std']      = df['iat'].rolling(20, min_periods=1).std().fillna(0)
    df['roll_entropy_mean'] = df['sig_entropy'].rolling(20, min_periods=1).mean()
    df['roll_hamming_mean'] = df['hamming_dist'].rolling(20, min_periods=1).mean()

    print('  [D] Encoding arb_id...')
    le = LabelEncoder()
    df['arb_id_enc'] = le.fit_transform(df['arb_id'].astype(str))
    return df

if os.path.exists(FEAT_PATH):
    print(f'Already exists — skipping: {FEAT_PATH}')
else:
    t0 = time.time()
    df = pd.read_parquet(CLEAN_PATH)
    print(f'Loaded: {df.shape} | attack={df["label"].mean()*100:.2f}%')

    MAX_FEAT = 2_000_000
    if len(df) > MAX_FEAT:
        print(f'Sampling {MAX_FEAT:,} from {len(df):,}...')
        df = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(frac=MAX_FEAT/len(df), random_state=42)
        ).reset_index(drop=True)
        print(f'After sample: {len(df):,} | attack={df["label"].mean()*100:.2f}%')

    fdf = engineer_features(df)
    del df; gc.collect()

    out = fdf[FEAT_COLS + ['label','attack_type']].fillna(0).replace([np.inf,-np.inf], 0)
    out.to_parquet(FEAT_PATH, index=False)
    print(f'\nSaved: {FEAT_PATH}  ({os.path.getsize(FEAT_PATH)/1e6:.1f} MB)  [{time.time()-t0:.1f}s]')
    print(f'Final shape: {out.shape} | attack={out["label"].mean()*100:.2f}%')
    del fdf, out; gc.collect()
ram()

# =============================================================================
#  STEP 5 — Train / Test Split with BALANCED classes
# =============================================================================
print('\n' + '='*68)
print('  STEP 5 — Train / Test Split (balanced)')
print('='*68)

fdf   = pd.read_parquet(FEAT_PATH)
clean = fdf[FEAT_COLS + ['label','attack_type']].dropna()
del fdf; gc.collect()

normal_df = clean[clean['label']==0]
attack_df = clean[clean['label']==1]
print(f'Before balance: normal={len(normal_df):,}  attack={len(attack_df):,}  ratio={len(normal_df)/max(len(attack_df),1):.1f}:1')

# Balance to 1:1 (equal normal and attack) for best model performance
n_each = min(len(normal_df), len(attack_df))
if n_each < 1000:
    raise RuntimeError(f'Too few attack samples ({n_each}). Check labeling in nb1.')

normal_bal = normal_df.sample(n_each, random_state=42)
attack_bal = attack_df.sample(n_each, random_state=42)
balanced   = pd.concat([normal_bal, attack_bal]).sample(frac=1, random_state=42)
del normal_df, attack_df, normal_bal, attack_bal; gc.collect()

print(f'After balance : {len(balanced):,} rows (1:1) | attack={balanced["label"].mean()*100:.1f}%')

X = balanced[FEAT_COLS].values.astype(np.float32)
y = balanced['label'].values.astype(np.int32)
g = balanced['attack_type'].values
del balanced; gc.collect()

X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
    X, y, g, test_size=0.3, random_state=42, stratify=y)
scaler  = RobustScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)
joblib.dump(scaler, os.path.join(OUT_DIR, 'scaler.pkl'))
print(f'Train: {len(X_tr):,}  Test: {len(X_te):,}')
print(f'Train attack rate: {y_tr.mean()*100:.1f}%')
ram()

# =============================================================================
#  STEP 6 — Evaluate Helper
# =============================================================================
all_results = {}
def evaluate(name, y_true, y_pred, y_proba):
    m = {
        'Accuracy' : round(accuracy_score(y_true, y_pred)*100, 2),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0)*100, 2),
        'Recall'   : round(recall_score(y_true, y_pred, zero_division=0)*100, 2),
        'F1'       : round(f1_score(y_true, y_pred, zero_division=0)*100, 2),
        'ROC-AUC'  : round(roc_auc_score(y_true, y_proba)*100, 2)
    }
    all_results[name] = m
    for k, v in m.items(): print(f'    {k:12s}: {v:.2f}%')
    return m

# =============================================================================
#  STEP 7 — Logistic Regression
# =============================================================================
print('\n' + '='*68); print('  STEP 7 — Logistic Regression'); print('='*68)
LR_PATH = os.path.join(OUT_DIR,'lr_model.pkl')
if os.path.exists(LR_PATH):
    print('Loading...'); lr = joblib.load(LR_PATH)
else:
    print('Training...')
    t0 = time.time()
    lr = LogisticRegression(C=0.6, max_iter=500, class_weight='balanced',
                             random_state=42, n_jobs=-1)
    lr.fit(X_tr_sc, y_tr); joblib.dump(lr, LR_PATH)
    print(f'Done in {time.time()-t0:.1f}s')
lr_pred  = lr.predict(X_te_sc); lr_proba = lr.predict_proba(X_te_sc)[:,1]
print('\n--- Logistic Regression ---')
evaluate('Logistic Regression', y_te, lr_pred, lr_proba)
print(classification_report(y_te, lr_pred, target_names=['Normal','Attack']))

# =============================================================================
#  STEP 8 — Random Forest
# =============================================================================
print('\n' + '='*68); print('  STEP 8 — Random Forest'); print('='*68)
RF_PATH = os.path.join(OUT_DIR,'rf_model.pkl')
if os.path.exists(RF_PATH):
    print('Loading...'); rf = joblib.load(RF_PATH)
else:
    print('Training...')
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                 max_features='sqrt', class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr); joblib.dump(rf, RF_PATH)
    print(f'Done in {time.time()-t0:.1f}s')
rf_pred  = rf.predict(X_te); rf_proba = rf.predict_proba(X_te)[:,1]
print('\n--- Random Forest ---')
evaluate('Random Forest', y_te, rf_pred, rf_proba)
print(classification_report(y_te, rf_pred, target_names=['Normal','Attack']))

# =============================================================================
#  STEP 9 — XGBoost
# =============================================================================
print('\n' + '='*68); print('  STEP 9 — XGBoost'); print('='*68)
XGB_PATH = os.path.join(OUT_DIR,'xgb_model.pkl')
if os.path.exists(XGB_PATH):
    print('Loading...'); xgb_m = joblib.load(XGB_PATH)
else:
    print('Training...')
    t0 = time.time()
    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=50)
    joblib.dump(xgb_m, XGB_PATH)
    print(f'Done in {time.time()-t0:.1f}s')
xgb_pred  = xgb_m.predict(X_te); xgb_proba = xgb_m.predict_proba(X_te)[:,1]
print('\n--- XGBoost ---')
evaluate('XGBoost', y_te, xgb_pred, xgb_proba)
print(classification_report(y_te, xgb_pred, target_names=['Normal','Attack']))

# =============================================================================
#  STEP 10 — MLP
# =============================================================================
print('\n' + '='*68); print('  STEP 10 — MLP'); print('='*68)
MLP_PATH = os.path.join(OUT_DIR,'mlp_model.pkl')
if os.path.exists(MLP_PATH):
    print('Loading...'); mlp = joblib.load(MLP_PATH)
else:
    print('Training...')
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(128,64,32), activation='relu', solver='adam',
                         alpha=0.01, learning_rate_init=1e-3, max_iter=200,
                         early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=15, random_state=42)
    mlp.fit(X_tr_sc, y_tr); joblib.dump(mlp, MLP_PATH)
    print(f'Stopped at iteration {mlp.n_iter_}  ({time.time()-t0:.1f}s)')
mlp_pred  = mlp.predict(X_te_sc); mlp_proba = mlp.predict_proba(X_te_sc)[:,1]
print('\n--- MLP ---')
evaluate('MLP Neural Network', y_te, mlp_pred, mlp_proba)
print(classification_report(y_te, mlp_pred, target_names=['Normal','Attack']))

# =============================================================================
#  STEP 11 — Results
# =============================================================================
print('\n' + '='*68)
print('  MODEL COMPARISON — ROAD DATASET')
print('='*68)
res_df = pd.DataFrame(all_results).T
res_df.to_csv(os.path.join(OUT_DIR,'model_results.csv'))
print(res_df.to_string())
print(f'\nBest: {res_df["Accuracy"].idxmax()} -> {res_df["Accuracy"].max():.2f}%')

# =============================================================================
#  STEP 12 — Plots
# =============================================================================
colors_m = ['#3498db','#2ecc71','#e74c3c','#9b59b6']
mlist    = list(all_results.keys())
mkeys    = ['Accuracy','Precision','Recall','F1','ROC-AUC']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Comparison - ROAD Dataset', fontsize=14, fontweight='bold')
x = np.arange(len(mkeys)); w = 0.18
for idx, (mn, col) in enumerate(zip(mlist, colors_m)):
    vals = [all_results[mn][k] for k in mkeys]
    axes[0].bar(x+idx*w, vals, w, label=mn, color=col, alpha=0.85)
axes[0].set_xticks(x+w*1.5); axes[0].set_xticklabels(mkeys, fontsize=9)
axes[0].set_ylim(0, 112); axes[0].set_ylabel('Score (%)'); axes[0].set_title('All Metrics')
axes[0].legend(fontsize=7); axes[0].axhline(88, color='gray', ls='--', lw=1)
axes[0].axhline(92, color='gray', ls=':', lw=1); axes[0].grid(axis='y', alpha=0.3)

roc_data = [('LR',lr_proba),('RF',rf_proba),('XGB',xgb_proba),('MLP',mlp_proba)]
for (mn, yp), col in zip(roc_data, colors_m):
    fpr, tpr, _ = roc_curve(y_te, yp)
    axes[1].plot(fpr, tpr, color=col, lw=2, label=f'{mn} ({roc_auc_score(y_te,yp)*100:.1f}%)')
axes[1].plot([0,1],[0,1],'k--',lw=1); axes[1].set_title('ROC Curves')
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

accs = {m: v['Accuracy'] for m,v in all_results.items()}
bars = axes[2].bar(list(accs.keys()), list(accs.values()), color=colors_m, alpha=0.85)
axes[2].axhspan(88, 92, color='yellow', alpha=0.25, label='Target 88-92%')
axes[2].set_ylim(50, 100); axes[2].set_ylabel('Accuracy (%)')
axes[2].set_title('Accuracy vs Target'); axes[2].legend(fontsize=8)
axes[2].tick_params(axis='x', rotation=15)
for bar, val in zip(bars, accs.values()):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)
axes[2].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.show(); print('Saved: model_comparison.png')

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
fig.suptitle('Confusion Matrices - ROAD', fontsize=13, fontweight='bold')
for ax, (name, yt, yp) in zip(axes, [
    ('LR',y_te,lr_pred),('RF',y_te,rf_pred),('XGB',y_te,xgb_pred),('MLP',y_te,mlp_pred)]):
    cm  = confusion_matrix(yt, yp)
    pct = cm.astype(float)/cm.sum()*100
    ann = np.array([[f'{v}\n({p:.1f}%)' for v,p in zip(rc,rp)] for rc,rp in zip(cm,pct)])
    sns.heatmap(cm, annot=ann, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Normal','Attack'], yticklabels=['Normal','Attack'],
                annot_kws={'size':9})
    ax.set_title(f'{name}\nacc={accuracy_score(yt,yp)*100:.1f}%', fontsize=10)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.show(); print('Saved: confusion_matrices.png')

fi = pd.Series(xgb_m.feature_importances_, index=FEAT_COLS).sort_values()
plt.figure(figsize=(10, 7))
fi.plot(kind='barh', color=['#e74c3c' if v>fi.quantile(0.75) else '#3498db' for v in fi])
plt.title('XGBoost Feature Importance - ROAD', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.axvline(fi.quantile(0.75), color='black', ls='--', lw=1, label='Top 25%')
plt.legend(fontsize=9); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show(); print('Saved: feature_importance.png')

type_rows = []
for atype in np.unique(g_te):
    mask = g_te == atype
    if mask.sum() < 10: continue
    yt, yp, ypr = y_te[mask], xgb_pred[mask], xgb_proba[mask]
    if len(np.unique(yt)) < 2: continue
    type_rows.append({'Attack Type':atype,'N':int(mask.sum()),
        'Accuracy':round(accuracy_score(yt,yp)*100,1),
        'Precision':round(precision_score(yt,yp,zero_division=0)*100,1),
        'Recall':round(recall_score(yt,yp,zero_division=0)*100,1),
        'F1':round(f1_score(yt,yp,zero_division=0)*100,1),
        'ROC-AUC':round(roc_auc_score(yt,ypr)*100,1)})
type_df = pd.DataFrame(type_rows)
type_df.to_csv(os.path.join(OUT_DIR,'per_attack_type_results.csv'), index=False)
print('\nXGBoost - Per Attack Type:')
print(type_df.to_string(index=False))

# =============================================================================
#  STEP 13 — Final Summary
# =============================================================================
print('\n' + '='*68)
print('  FINAL SUMMARY - ROAD DATASET')
print('='*68)
print(f'  Dataset      : ROAD (ORNL)')
print(f'  Features     : {len(FEAT_COLS)}')
print(f'  Train samples: {len(X_tr):,}')
print(f'  Test samples : {len(X_te):,}')
for m, r in res_df.iterrows():
    acc = r['Accuracy']
    tag = '[TARGET]' if 88<=acc<=92 else ('[ABOVE]' if acc>92 else '[BELOW]')
    print(f'  {m:<25s}  {acc:>9.2f}%  F1={r["F1"]:>6.2f}%  {tag}')
print(f'\n  Output: {OUT_DIR}')
print('='*68)
ram()
print('\nNB1 complete. Delete road_output/ and rerun if attack counts were 0.')
print('Then run nb2_otids.py, then nb4_cross_vehicle.py')