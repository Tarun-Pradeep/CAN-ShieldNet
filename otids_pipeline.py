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

# =============================================================================
#  SET THESE PATHS
# =============================================================================
OTIDS_DIR = '/Users/tarun/Desktop/can3/10) CAN-Intrusion Dataset'
OUT_DIR   = '/Users/tarun/Desktop/otids_output'
# =============================================================================

os.makedirs(OUT_DIR, exist_ok=True)

RAW_PATH   = os.path.join(OUT_DIR, 'otids_raw.parquet')
CLEAN_PATH = os.path.join(OUT_DIR, 'otids_cleaned.parquet')
FEAT_PATH  = os.path.join(OUT_DIR, 'otids_features.parquet')

FEAT_COLS = [
    'byte_0','byte_1','byte_2','byte_3','byte_4','byte_5','byte_6','byte_7',
    'byte_mean','byte_std','byte_min','byte_max','byte_range','byte_entropy',
    'iat','iat_zscore','freq_dev','hamming','dlc_change',
    'roll_iat_mean','roll_iat_std','roll_entropy_mean',
    'arb_id_enc'
]

FILE_TYPE_MAP = {
    'attack_free'   : 'Normal',
    'dos'           : 'DoS',
    'fuzzy'         : 'Fuzzy',
    'impersonation' : 'Impersonation',
}

OTIDS_RE = re.compile(r'Timestamp:\s+([\d.]+)\s+ID:\s+(\w+)\s+\w+\s+DLC:\s+(\d+)\s+(.*)')
MAX_ROWS = 600_000

def ram():
    p = psutil.virtual_memory()
    print(f'  RAM  used={(p.total-p.available)/1e9:.1f}GB  avail={p.available/1e9:.1f}GB')

def get_otids_type(fname):
    fl = fname.lower()
    return next((v for k,v in FILE_TYPE_MAP.items() if k in fl), 'Normal')

def assign_label(df, attack_type):
    if attack_type == 'Normal':
        df['label'] = 0
    elif attack_type == 'DoS':
        dos_mask = df['arb_id'] == '0000'
        if dos_mask.any():
            onset = df.loc[dos_mask, 'timestamp'].iloc[0]
            df['label'] = ((df['timestamp'] >= onset) & dos_mask).astype(np.int8)
        else:
            mid = df['timestamp'].median()
            df['label'] = (df['timestamp'] > mid).astype(np.int8)
    elif attack_type in ('Fuzzy', 'Impersonation'):
        t0 = df['timestamp'].min(); tr = df['timestamp'].max() - t0
        onset = t0 + (300.0 if tr > 600.0 else tr * 0.5)
        df['label'] = (df['timestamp'] > onset).astype(np.int8)
    return df

def parse_otids_file(fpath, attack_type):
    rows = []
    with open(fpath, 'r', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= MAX_ROWS: break
            m = OTIDS_RE.search(line)
            if m:
                rows.append((float(m.group(1)), m.group(2).strip().lower(),
                             int(m.group(3)), m.group(4).strip()))
    if not rows:
        print(f'  WARNING: 0 rows matched in {os.path.basename(fpath)}')
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['timestamp','arb_id','dlc','data'])
    df = assign_label(df, attack_type)
    df['attack_type'] = attack_type
    return df[['timestamp','arb_id','dlc','data','label','attack_type']]

# =============================================================================
#  STEP 1 — Find .txt files
# =============================================================================
print('='*68)
print('  STEP 1 — Finding OTIDS .txt files')
print('='*68)

txt_files = glob.glob(os.path.join(OTIDS_DIR, '*.txt'))
if not txt_files:
    # search subdirs
    txt_files = glob.glob(os.path.join(OTIDS_DIR, '**', '*.txt'), recursive=True)

if not txt_files:
    raise FileNotFoundError(f'No .txt files found in {OTIDS_DIR}')

print(f'Found {len(txt_files)} .txt files:')
for f in sorted(txt_files):
    print(f'  {os.path.basename(f):40s} -> {get_otids_type(os.path.basename(f))}')
ram()

# =============================================================================
#  STEP 2 — Parse all .txt files → Raw Parquet
# =============================================================================
print('\n' + '='*68)
print('  STEP 2 — Parsing .txt files')
print('='*68)

if os.path.exists(RAW_PATH):
    print(f'Already exists — skipping: {RAW_PATH}')
else:
    all_frames = []
    for fpath in sorted(txt_files):
        atype = get_otids_type(os.path.basename(fpath))
        print(f'\n  [{atype}]  {os.path.basename(fpath)}')
        t0 = time.time()
        df = parse_otids_file(fpath, atype)
        if len(df) > 0:
            all_frames.append(df)
            print(f'    {len(df):,} rows | attack={df["label"].mean()*100:.1f}% | IDs={df["arb_id"].nunique()}  ({time.time()-t0:.1f}s)')
        else:
            print(f'    Empty — skipped')
        ram()

    raw = pd.concat(all_frames, ignore_index=True)
    raw.to_parquet(RAW_PATH, index=False)
    print(f'\nSaved: {RAW_PATH}')
    print(f'  Total rows : {len(raw):,}')
    print(f'  Attack rows: {raw["label"].sum():,}  ({raw["label"].mean()*100:.2f}%)')
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
    b = len(df); df.dropna(subset=['timestamp'], inplace=True); print(f'[1] Drop null timestamps: -{b-len(df):,}')
    b = len(df); df = df[df['timestamp'] > 0]; print(f'[2] Remove ts<=0: -{b-len(df):,}')
    b = len(df); df = df[df['arb_id'].notna() & (df['arb_id'].astype(str).str.strip() != '')]; print(f'[3] Drop empty arb_id: -{b-len(df):,}')
    b = len(df); df['label'] = pd.to_numeric(df['label'], errors='coerce'); df = df[df['label'].isin([0,1])]; df['label'] = df['label'].astype(np.int8); print(f'[4] Binary label: -{b-len(df):,}')
    b = len(df); df = df[df['dlc'].between(0, 8)]; print(f'[5] DLC 0-8: -{b-len(df):,}')
    b = len(df); df.drop_duplicates(subset=['timestamp','arb_id','data'], keep='first', inplace=True); print(f'[6] Dedup: -{b-len(df):,}')
    df['arb_id'] = df['arb_id'].astype(str).str.strip().str.lower()
    df.sort_values(['attack_type','timestamp'], inplace=True); df.reset_index(drop=True, inplace=True)
    print(f'Final rows: {len(df):,} | attack={df["label"].mean()*100:.2f}%')
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

def parse_data_bytes(data_str, n=8):
    try:
        tokens = str(data_str).strip().split()
        bvals  = [int(t, 16) for t in tokens[:n]]
        while len(bvals) < n: bvals.append(0)
        return bvals
    except: return [0]*n

def byte_entropy(blist):
    a = np.array(blist, dtype=np.float32)
    if a.sum() == 0: return 0.0
    _, c = np.unique(a, return_counts=True)
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))

def hamming_dist(b1, b2):
    return sum(bin(x^y).count('1') for x,y in zip(b1,b2))

def engineer_features(df, window=20):
    df   = df.copy().sort_values('timestamp').reset_index(drop=True)
    n    = len(df)

    print('  [A] Parsing bytes + stats...')
    bmat = np.array([parse_data_bytes(d) for d in df['data']], dtype=np.int16)
    for i in range(8): df[f'byte_{i}'] = bmat[:,i]
    df['byte_mean']    = bmat.mean(axis=1)
    df['byte_std']     = bmat.std(axis=1)
    df['byte_min']     = bmat.min(axis=1)
    df['byte_max']     = bmat.max(axis=1)
    df['byte_range']   = df['byte_max'] - df['byte_min']
    df['byte_entropy'] = [byte_entropy(bmat[i]) for i in range(n)]

    print('  [B] Per-ID temporal features...')
    iat_arr  = np.zeros(n, np.float32)
    ham_arr  = np.zeros(n, np.float32)
    dlcc_arr = np.zeros(n, np.float32)
    lt, lb, ld = {}, {}, {}
    for i in range(n):
        cid = df.at[i,'arb_id']; ts = df.at[i,'timestamp']; dlc = df.at[i,'dlc']; bv = bmat[i].tolist()
        if cid in lt:
            iat_arr[i]  = ts - lt[cid]
            ham_arr[i]  = hamming_dist(bv, lb[cid])
            dlcc_arr[i] = float(dlc != ld[cid])
        lt[cid] = ts; lb[cid] = bv; ld[cid] = dlc
    df['iat'] = iat_arr; df['hamming'] = ham_arr; df['dlc_change'] = dlcc_arr
    df['iat_zscore'] = df.groupby('arb_id')['iat'].transform(lambda x: (x-x.mean())/(x.std()+1e-9))
    df['iat_inv']    = 1.0 / df['iat'].clip(lower=1e-6)
    df['freq_dev']   = df.groupby('arb_id')['iat_inv'].transform(lambda x: (x-x.mean())/(x.std()+1e-9))
    df.drop(columns=['iat_inv'], inplace=True)

    print('  [C] Rolling window...')
    df['roll_iat_mean']     = df['iat'].rolling(window, min_periods=1).mean()
    df['roll_iat_std']      = df['iat'].rolling(window, min_periods=1).std().fillna(0)
    df['roll_entropy_mean'] = df['byte_entropy'].rolling(window, min_periods=1).mean()

    print('  [D] Encoding arb_id...')
    le = LabelEncoder()
    df['arb_id_enc'] = le.fit_transform(df['arb_id'].astype(str))
    return df

if os.path.exists(FEAT_PATH):
    print(f'Already exists — skipping: {FEAT_PATH}')
else:
    t0 = time.time()
    df = pd.read_parquet(CLEAN_PATH)

    MAX_FEAT = 2_000_000
    if len(df) > MAX_FEAT:
        print(f'Sampling {MAX_FEAT:,} rows from {len(df):,}...')
        df = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(frac=MAX_FEAT/len(df), random_state=42)
        ).reset_index(drop=True)

    fdf = engineer_features(df)
    del df; gc.collect()
    fdf[FEAT_COLS + ['label','attack_type']].fillna(0).to_parquet(FEAT_PATH, index=False)
    print(f'Saved: {FEAT_PATH}  ({os.path.getsize(FEAT_PATH)/1e6:.1f} MB)  [{time.time()-t0:.1f}s]')
    del fdf; gc.collect()
ram()

# =============================================================================
#  STEP 5 — Train / Test Split
# =============================================================================
print('\n' + '='*68)
print('  STEP 5 — Train / Test Split')
print('='*68)

fdf   = pd.read_parquet(FEAT_PATH)
clean = fdf[FEAT_COLS + ['label','attack_type']].dropna()
del fdf; gc.collect()

X = clean[FEAT_COLS].values.astype(np.float32)
y = clean['label'].values.astype(np.int32)
g = clean['attack_type'].values
del clean; gc.collect()

X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(X, y, g, test_size=0.3, random_state=42, stratify=y)
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
    for k,v in m.items(): print(f'    {k:12s}: {v:.2f}%')
    return m

# =============================================================================
#  STEP 7 — Logistic Regression
# =============================================================================
print('\n' + '='*68); print('  STEP 7 — Logistic Regression'); print('='*68)
LR_PATH = os.path.join(OUT_DIR,'lr_model.pkl')
if os.path.exists(LR_PATH): print('Loading...'); lr = joblib.load(LR_PATH)
else:
    print('Training...')
    t0 = time.time()
    lr = LogisticRegression(C=0.8, max_iter=500, class_weight='balanced', random_state=42, n_jobs=-1)
    lr.fit(X_tr_sc, y_tr); joblib.dump(lr, LR_PATH); print(f'Done in {time.time()-t0:.1f}s')
lr_pred = lr.predict(X_te_sc); lr_proba = lr.predict_proba(X_te_sc)[:,1]
print('\n--- Logistic Regression ---'); evaluate('Logistic Regression', y_te, lr_pred, lr_proba)
print(classification_report(y_te, lr_pred, target_names=['Normal','Attack']))

# =============================================================================
#  STEP 8 — Random Forest
# =============================================================================
print('\n' + '='*68); print('  STEP 8 — Random Forest'); print('='*68)
RF_PATH = os.path.join(OUT_DIR,'rf_model.pkl')
if os.path.exists(RF_PATH): print('Loading...'); rf = joblib.load(RF_PATH)
else:
    print('Training...')
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=4,
                                 max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr); joblib.dump(rf, RF_PATH); print(f'Done in {time.time()-t0:.1f}s')
rf_pred = rf.predict(X_te); rf_proba = rf.predict_proba(X_te)[:,1]
print('\n--- Random Forest ---'); evaluate('Random Forest', y_te, rf_pred, rf_proba)
print(classification_report(y_te, rf_pred, target_names=['Normal','Attack']))

# =============================================================================
#  STEP 9 — XGBoost
# =============================================================================
print('\n' + '='*68); print('  STEP 9 — XGBoost'); print('='*68)
XGB_PATH = os.path.join(OUT_DIR,'xgb_model.pkl')
spw = (y_tr==0).sum() / max((y_tr==1).sum(), 1)
if os.path.exists(XGB_PATH): print('Loading...'); xgb_m = joblib.load(XGB_PATH)
else:
    print('Training...')
    t0 = time.time()
    xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                scale_pos_weight=spw, eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=50)
    joblib.dump(xgb_m, XGB_PATH); print(f'Done in {time.time()-t0:.1f}s')
xgb_pred = xgb_m.predict(X_te); xgb_proba = xgb_m.predict_proba(X_te)[:,1]
print('\n--- XGBoost ---'); evaluate('XGBoost', y_te, xgb_pred, xgb_proba)
print(classification_report(y_te, xgb_pred, target_names=['Normal','Attack']))

# =============================================================================
#  STEP 10 — MLP
# =============================================================================
print('\n' + '='*68); print('  STEP 10 — MLP Neural Network'); print('='*68)
MLP_PATH = os.path.join(OUT_DIR,'mlp_model.pkl')
if os.path.exists(MLP_PATH): print('Loading...'); mlp = joblib.load(MLP_PATH)
else:
    print('Training...')
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(128,64,32), activation='relu', solver='adam',
                         alpha=0.005, learning_rate_init=1e-3, max_iter=200,
                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, random_state=42)
    mlp.fit(X_tr_sc, y_tr); joblib.dump(mlp, MLP_PATH)
    print(f'Stopped at iteration {mlp.n_iter_}  ({time.time()-t0:.1f}s)')
mlp_pred = mlp.predict(X_te_sc); mlp_proba = mlp.predict_proba(X_te_sc)[:,1]
print('\n--- MLP Neural Network ---'); evaluate('MLP Neural Network', y_te, mlp_pred, mlp_proba)
print(classification_report(y_te, mlp_pred, target_names=['Normal','Attack']))

# =============================================================================
#  STEP 11 — Results Table
# =============================================================================
print('\n' + '='*68)
print('  MODEL COMPARISON — OTIDS (CAN-Intrusion) 70/30 split')
print('='*68)
res_df = pd.DataFrame(all_results).T
res_df.to_csv(os.path.join(OUT_DIR,'model_results.csv'))
print(res_df.to_string())
print(f'\nBest: {res_df["Accuracy"].idxmax()}  ->  {res_df["Accuracy"].max():.2f}%')
for m, row in res_df.iterrows():
    acc = row['Accuracy']
    tag = '[IN TARGET]' if 90<=acc<=95 else ('[ABOVE]' if acc>95 else '[BELOW]')
    print(f'  {m:<25s}: {acc:.2f}%  {tag}')

# =============================================================================
#  STEP 12 — Plots
# =============================================================================
colors_m = ['#3498db','#2ecc71','#e74c3c','#9b59b6']
mlist    = list(all_results.keys())
mkeys    = ['Accuracy','Precision','Recall','F1','ROC-AUC']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Comparison - OTIDS Dataset', fontsize=14, fontweight='bold')
x = np.arange(len(mkeys)); w = 0.18
for idx, (mn, col) in enumerate(zip(mlist, colors_m)):
    vals = [all_results[mn][k] for k in mkeys]
    axes[0].bar(x+idx*w, vals, w, label=mn, color=col, alpha=0.85)
axes[0].set_xticks(x+w*1.5); axes[0].set_xticklabels(mkeys, fontsize=9)
axes[0].set_ylim(0, 112); axes[0].set_ylabel('Score (%)'); axes[0].set_title('All Metrics')
axes[0].legend(fontsize=7); axes[0].axhline(90, color='gray', ls='--', lw=1)
axes[0].axhline(95, color='gray', ls=':', lw=1); axes[0].grid(axis='y', alpha=0.3)

roc_data = [('LR',lr_proba),('RF',rf_proba),('XGB',xgb_proba),('MLP',mlp_proba)]
for (mn,yp),col in zip(roc_data, colors_m):
    fpr,tpr,_ = roc_curve(y_te, yp)
    axes[1].plot(fpr, tpr, color=col, lw=2, label=f'{mn} ({roc_auc_score(y_te,yp)*100:.1f}%)')
axes[1].plot([0,1],[0,1],'k--',lw=1); axes[1].set_title('ROC Curves')
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

accs = {m: v['Accuracy'] for m,v in all_results.items()}
bars = axes[2].bar(list(accs.keys()), list(accs.values()), color=colors_m, alpha=0.85)
axes[2].axhspan(90, 95, color='yellow', alpha=0.25, label='Target 90-95%')
axes[2].set_ylim(60, 100); axes[2].set_ylabel('Accuracy (%)'); axes[2].set_title('Accuracy vs Target')
axes[2].legend(fontsize=8); axes[2].tick_params(axis='x', rotation=15)
for bar, val in zip(bars, accs.values()):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)
axes[2].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'model_comparison.png'), dpi=150, bbox_inches='tight'); plt.show()
print('Saved: model_comparison.png')

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
fig.suptitle('Confusion Matrices - OTIDS', fontsize=13, fontweight='bold')
for ax, (name, yt, yp) in zip(axes, [('LR',y_te,lr_pred),('RF',y_te,rf_pred),('XGB',y_te,xgb_pred),('MLP',y_te,mlp_pred)]):
    cm  = confusion_matrix(yt, yp)
    pct = cm.astype(float)/cm.sum()*100
    ann = np.array([[f'{v}\n({p:.1f}%)' for v,p in zip(rc,rp)] for rc,rp in zip(cm,pct)])
    sns.heatmap(cm, annot=ann, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Normal','Attack'], yticklabels=['Normal','Attack'], annot_kws={'size':9})
    ax.set_title(f'{name}\nacc={accuracy_score(yt,yp)*100:.1f}%', fontsize=10)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'confusion_matrices.png'), dpi=150, bbox_inches='tight'); plt.show()
print('Saved: confusion_matrices.png')

fi = pd.Series(xgb_m.feature_importances_, index=FEAT_COLS).sort_values()
plt.figure(figsize=(10, 8))
fi.plot(kind='barh', color=['#e74c3c' if v>fi.quantile(0.75) else '#3498db' for v in fi])
plt.title('XGBoost Feature Importance - OTIDS', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.axvline(fi.quantile(0.75), color='black', ls='--', lw=1, label='Top 25%')
plt.legend(fontsize=9); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'feature_importance.png'), dpi=150, bbox_inches='tight'); plt.show()
print('Saved: feature_importance.png')

type_rows = []
for atype in np.unique(g_te):
    mask = g_te == atype
    if mask.sum() < 20: continue
    yt, yp, ypr = y_te[mask], xgb_pred[mask], xgb_proba[mask]
    if len(np.unique(yt)) < 2: continue
    type_rows.append({'Attack Type':atype,'N':int(mask.sum()),
        'Accuracy':round(accuracy_score(yt,yp)*100,1),'Precision':round(precision_score(yt,yp,zero_division=0)*100,1),
        'Recall':round(recall_score(yt,yp,zero_division=0)*100,1),'F1':round(f1_score(yt,yp,zero_division=0)*100,1),
        'ROC-AUC':round(roc_auc_score(yt,ypr)*100,1)})
type_df = pd.DataFrame(type_rows)
type_df.to_csv(os.path.join(OUT_DIR,'per_attack_type_results.csv'), index=False)
print('\nXGBoost - Per Attack Type:'); print(type_df.to_string(index=False))

# =============================================================================
#  STEP 13 — Final Summary
# =============================================================================
print('\n' + '='*68)
print('  FINAL SUMMARY - OTIDS DATASET')
print('='*68)
print(f'  Dataset      : CAN-Intrusion (HCRL / OTIDS)')
print(f'  Features     : {len(FEAT_COLS)}')
print(f'  Train samples: {len(X_tr):,}')
print(f'  Test samples : {len(X_te):,}')
print()
for m, r in res_df.iterrows():
    acc = r['Accuracy']
    tag = '[TARGET]' if 90<=acc<=95 else ('[ABOVE]' if acc>95 else '[BELOW]')
    print(f'  {m:<25s}  {acc:>9.2f}%  {r["F1"]:>7.2f}%  {r["ROC-AUC"]:>9.2f}%  {tag}')
print(f'\n  Output saved to: {OUT_DIR}')
print('='*68)
ram()
print('\nNB2 complete. Run nb4_cross_vehicle.py next.')