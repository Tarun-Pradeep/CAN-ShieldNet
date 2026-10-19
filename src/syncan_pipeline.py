import os, gc, glob, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil, joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
import xgboost as xgb

warnings.filterwarnings('ignore')

SYNCAN_DIR = 'data/syncan'
OUT_DIR    = 'outputs/syncan'
os.makedirs(OUT_DIR, exist_ok=True)

FEAT_COLS = [
    'arb_id_enc',
    'data_len',
    'sig_0','sig_1','sig_2','sig_3',
    'sig_mean','sig_std','sig_min','sig_max','sig_range',
    'sig_entropy','sig_zero_count',
    'sig_delta_mean','sig_delta_max','payload_changed',
    'iat','iat_zscore','freq_dev','iat_ratio','burst_flag',
    'roll_iat_mean','roll_iat_std','roll_entropy_mean','roll_delta_mean',
]

# Attack type map — checks folder name
ATTACK_MAP = {
    'attack-free'       : 'Normal',
    'extra-attack-free' : 'Normal',
    'fuzzing'           : 'Fuzzy_Injection',
    'dos'               : 'DoS',
    'interval'          : 'Interval',
    'rpm'               : 'RPM',
    'speed'             : 'Speed',
    'standstill'        : 'Standstill',
    'systematic'        : 'Systematic',
    'gear'              : 'Gear_Attack',
    'combined'          : 'Combined_Attack',
    'accessory'         : 'Accessory',
    'continuous'        : 'Continuous_Drift',
    'fuzzy'             : 'Fuzzy_Injection',
    'plateau'           : 'Plateau_Attack',
    'playback'          : 'Playback_Attack',
    'suppress'          : 'Suppression',
    'flooding'          : 'Flooding',
}

def ram():
    p = psutil.virtual_memory()
    print(f'  RAM  used={(p.total-p.available)/1e9:.1f}GB  avail={p.available/1e9:.1f}GB')

def get_attack_type(fpath):
    parts = fpath.replace('\\', '/').lower().split('/')
    # Check from deepest folder upward
    for part in reversed(parts[:-1]):  # skip filename
        for k, v in ATTACK_MAP.items():
            if k in part:
                return v
    return 'Unknown'

def decode_hex_row(h):
    """Decode hex string to 4 float32 signal values + length feature."""
    try:
        s = str(h).strip()
        # Pad to 16 hex chars (8 bytes) on the RIGHT
        padded = s.ljust(16, '0')[:16]
        b = bytes.fromhex(padded)
        return (
            len(s) // 2,                          # data_len (actual byte count)
            float(b[0] << 8 | b[1]),              # sig_0
            float(b[2] << 8 | b[3]),              # sig_1
            float(b[4] << 8 | b[5]),              # sig_2
            float(b[6] << 8 | b[7]),              # sig_3
        )
    except Exception:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

def signal_entropy(vals):
    a = np.array(vals, dtype=np.float32)
    if np.all(a == 0): return 0.0
    _, c = np.unique(np.round(a, 2), return_counts=True)
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))

# =============================================================================
#  STEP 1 — Find CSV files
# =============================================================================
print('='*68)
print('  STEP 1 — Finding CSV files')
print('='*68)

csv_files = sorted(glob.glob(os.path.join(SYNCAN_DIR, '**', '*.csv'), recursive=True))
if not csv_files:
    raise FileNotFoundError(f'No CSV files found in {SYNCAN_DIR}')

print(f'Found {len(csv_files)} CSV files')
# Show sample
for f in csv_files[:8]:
    print(f'  {os.path.relpath(f, SYNCAN_DIR):60s} -> {get_attack_type(f)}')
if len(csv_files) > 8:
    print(f'  ... and {len(csv_files)-8} more')
ram()

# =============================================================================
#  STEP 2 — Load & Parse
# =============================================================================
print('\n' + '='*68)
print('  STEP 2 — Loading CSV files')
print('='*68)

ROWS_PER_FILE = 50_000   # cap per file to keep RAM safe
normal_dfs  = []
attack_dfs  = []
t0 = time.time()

for fpath in csv_files:
    atype = get_attack_type(fpath)
    is_normal = (atype == 'Normal')
    try:
        df = pd.read_csv(fpath, low_memory=False, on_bad_lines='skip',
                         nrows=ROWS_PER_FILE)
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Timestamp
        if 'timestamp' in df.columns:
            df['time'] = pd.to_numeric(df['timestamp'], errors='coerce')
        else:
            df['time'] = np.arange(len(df), dtype=np.float64)
        df.dropna(subset=['time'], inplace=True)

        # Arb ID
        if 'arbitration_id' in df.columns:
            df['arb_id'] = df['arbitration_id'].astype(str).str.strip().str.upper()
        elif 'id' in df.columns:
            df['arb_id'] = df['id'].astype(str).str.strip().str.upper()
        else:
            df['arb_id'] = 'UNKNOWN'

        # Decode hex data_field
        if 'data_field' in df.columns:
            decoded = df['data_field'].apply(decode_hex_row)
            df['data_len'] = decoded.apply(lambda x: x[0]).astype(np.float32)
            df['sig_0']    = decoded.apply(lambda x: x[1]).astype(np.float32)
            df['sig_1']    = decoded.apply(lambda x: x[2]).astype(np.float32)
            df['sig_2']    = decoded.apply(lambda x: x[3]).astype(np.float32)
            df['sig_3']    = decoded.apply(lambda x: x[4]).astype(np.float32)
        else:
            df['data_len'] = df['sig_0'] = df['sig_1'] = df['sig_2'] = df['sig_3'] = np.float32(0)

        df['label']       = np.int8(0 if is_normal else 1)
        df['attack_type'] = atype

        keep = ['time','arb_id','label','attack_type',
                'data_len','sig_0','sig_1','sig_2','sig_3']
        df = df[keep]

        if is_normal:
            normal_dfs.append(df)
        else:
            attack_dfs.append(df)

        print(f'  {"N" if is_normal else "A"}  {os.path.relpath(fpath, SYNCAN_DIR):55s}  {len(df):>7,} rows')
    except Exception as e:
        print(f'  ERR {os.path.relpath(fpath, SYNCAN_DIR):55s}  {e}')

print(f'\nConcatenating...')
normal_df = pd.concat(normal_dfs, ignore_index=True) if normal_dfs else pd.DataFrame()
attack_df = pd.concat(attack_dfs, ignore_index=True) if attack_dfs else pd.DataFrame()
del normal_dfs, attack_dfs; gc.collect()

print(f'Normal rows : {len(normal_df):,}')
print(f'Attack rows : {len(attack_df):,}')

# Balance classes — sample equal numbers
N = min(len(normal_df), len(attack_df), 2_000_000)
print(f'\nBalancing to {N:,} rows per class...')
normal_df = normal_df.sample(n=N, random_state=42)
attack_df = attack_df.sample(n=N, random_state=42)

raw = pd.concat([normal_df, attack_df], ignore_index=True)
del normal_df, attack_df; gc.collect()

raw = raw.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
print(f'Total balanced rows: {len(raw):,}')
print(f'Attack rate: {raw["label"].mean()*100:.2f}%')
print(f'Loaded in {time.time()-t0:.1f}s')
ram()

# =============================================================================
#  STEP 3 — Clean
# =============================================================================
print('\n' + '='*68)
print('  STEP 3 — Cleaning')
print('='*68)

before = len(raw)
raw = raw[raw['time'] > 0]
raw = raw[raw['arb_id'].notna() & (raw['arb_id'].str.strip() != '')]
raw['label'] = raw['label'].astype(np.int8)

sig_cols = ['sig_0','sig_1','sig_2','sig_3']
for col in sig_cols:
    raw[col] = pd.to_numeric(raw[col], errors='coerce').fillna(0.0)
    m, s = raw[col].mean(), raw[col].std()
    if s > 0:
        raw[col] = raw[col].clip(m - 4*s, m + 4*s)

raw = raw.reset_index(drop=True)
print(f'Before: {before:,}  After: {len(raw):,}')
ram()

# =============================================================================
#  STEP 4 — Feature Engineering
# =============================================================================
print('\n' + '='*68)
print('  STEP 4 — Feature Engineering')
print('='*68)

t0 = time.time()
df = raw.sort_values('time').reset_index(drop=True)
del raw; gc.collect()

n    = len(df)
smat = df[['sig_0','sig_1','sig_2','sig_3']].values.astype(np.float32)

# Group A — Signal statistics
print('  [A] Signal statistics...')
df['sig_mean']       = smat.mean(axis=1)
df['sig_std']        = smat.std(axis=1)
df['sig_min']        = smat.min(axis=1)
df['sig_max']        = smat.max(axis=1)
df['sig_range']      = df['sig_max'] - df['sig_min']
df['sig_entropy']    = [signal_entropy(smat[i]) for i in range(n)]
df['sig_zero_count'] = (smat == 0).sum(axis=1).astype(np.float32)

# Group B — Signal deltas per arb_id
print('  [B] Signal deltas...')
d_mean   = np.zeros(n, np.float32)
d_max    = np.zeros(n, np.float32)
pchg     = np.zeros(n, np.float32)
last_sig = {}
for i in range(n):
    cid = df.at[i, 'arb_id']
    sv  = smat[i]
    if cid in last_sig:
        delta     = np.abs(sv - last_sig[cid])
        d_mean[i] = delta.mean()
        d_max[i]  = delta.max()
        pchg[i]   = float(np.any(delta > 0.01))
    last_sig[cid] = sv
df['sig_delta_mean']  = d_mean
df['sig_delta_max']   = d_max
df['payload_changed'] = pchg

# Group C — IAT / temporal
print('  [C] IAT / temporal...')
iat_arr   = np.zeros(n, np.float32)
burst_arr = np.zeros(n, np.float32)
last_time = {}
iat_hist  = {}
for i in range(n):
    cid = df.at[i, 'arb_id']
    ts  = df.at[i, 'time']
    if cid in last_time:
        iat = ts - last_time[cid]
        iat_arr[i] = max(iat, 0.0)
        hist = iat_hist.get(cid, [])
        if len(hist) >= 3:
            med = np.median(hist[-5:])
            burst_arr[i] = float(iat < 0.2 * med) if med > 0 else 0.0
        iat_hist.setdefault(cid, []).append(iat)
    last_time[cid] = ts

df['iat']        = iat_arr
df['burst_flag'] = burst_arr
df['iat_zscore'] = df.groupby('arb_id')['iat'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-9))
df['iat_inv']   = 1.0 / df['iat'].clip(lower=1e-6)
df['freq_dev']  = df.groupby('arb_id')['iat_inv'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-9))
df['iat_ratio'] = df['iat'] / (df['iat'].mean() + 1e-9)
df.drop(columns=['iat_inv'], inplace=True)

# Group D — Rolling window
print('  [D] Rolling window...')
df['roll_iat_mean']     = df['iat'].rolling(20, min_periods=1).mean()
df['roll_iat_std']      = df['iat'].rolling(20, min_periods=1).std().fillna(0)
df['roll_entropy_mean'] = df['sig_entropy'].rolling(20, min_periods=1).mean()
df['roll_delta_mean']   = df['sig_delta_mean'].rolling(20, min_periods=1).mean()

# Group E — Encode arb_id
print('  [E] Encoding arb_id...')
le = LabelEncoder()
df['arb_id_enc'] = le.fit_transform(df['arb_id'].astype(str))

print(f'  Done in {time.time()-t0:.1f}s  |  Shape: {df.shape}')
ram()

# =============================================================================
#  STEP 5 — Clean NaNs & Split
# =============================================================================
print('\n' + '='*68)
print('  STEP 5 — Train / Test Split')
print('='*68)

df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0).replace([np.inf, -np.inf], 0.0)

X = df[FEAT_COLS].values.astype(np.float32)
y = df['label'].values
g = df['attack_type'].values
del df; gc.collect()

X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
    X, y, g, test_size=0.3, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)
joblib.dump(scaler, os.path.join(OUT_DIR, 'scaler.pkl'))

print(f'Train : {X_tr_sc.shape}  |  Attack rate: {y_tr.mean()*100:.2f}%')
print(f'Test  : {X_te_sc.shape}  |  Attack rate: {y_te.mean()*100:.2f}%')
ram()

# =============================================================================
#  STEP 6 — Evaluation Helper
# =============================================================================
all_results = {}

def evaluate(name, y_true, y_pred, y_proba):
    acc  = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec  = recall_score(y_true, y_pred, zero_division=0) * 100
    f1   = f1_score(y_true, y_pred, zero_division=0) * 100
    auc  = roc_auc_score(y_true, y_proba) * 100
    all_results[name] = {
        'Accuracy' : round(acc,  2),
        'Precision': round(prec, 2),
        'Recall'   : round(rec,  2),
        'F1'       : round(f1,   2),
        'ROC-AUC'  : round(auc,  2),
    }
    print(f'  Accuracy : {acc:.2f}%')
    print(f'  Precision: {prec:.2f}%')
    print(f'  Recall   : {rec:.2f}%')
    print(f'  F1       : {f1:.2f}%')
    print(f'  ROC-AUC  : {auc:.2f}%')

# =============================================================================
#  STEP 7 — Logistic Regression
# =============================================================================
print('\n' + '='*68)
print('  STEP 7 — Logistic Regression')
print('='*68)

LR_PATH = os.path.join(OUT_DIR, 'lr_model.pkl')
if os.path.exists(LR_PATH):
    print('Loading saved LR...')
    lr = joblib.load(LR_PATH)
else:
    print('Training...')
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                             random_state=42, n_jobs=-1)
    lr.fit(X_tr_sc, y_tr)
    joblib.dump(lr, LR_PATH)
    print(f'Done in {time.time()-t0:.1f}s')

lr_pred  = lr.predict(X_te_sc)
lr_proba = lr.predict_proba(X_te_sc)[:, 1]
print('\n--- Logistic Regression ---')
evaluate('Logistic Regression', y_te, lr_pred, lr_proba)
print(classification_report(y_te, lr_pred, target_names=['Normal','Attack']))

# =============================================================================
#  STEP 8 — Random Forest
# =============================================================================
print('\n' + '='*68)
print('  STEP 8 — Random Forest')
print('='*68)

RF_PATH = os.path.join(OUT_DIR, 'rf_model.pkl')
if os.path.exists(RF_PATH):
    print('Loading saved RF...')
    rf = joblib.load(RF_PATH)
else:
    print('Training...')
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_tr_sc, y_tr)
    joblib.dump(rf, RF_PATH)
    print(f'Done in {time.time()-t0:.1f}s')

rf_pred  = rf.predict(X_te_sc)
rf_proba = rf.predict_proba(X_te_sc)[:, 1]
print('\n--- Random Forest ---')
evaluate('Random Forest', y_te, rf_pred, rf_proba)
print(classification_report(y_te, rf_pred, target_names=['Normal','Attack']))

# =============================================================================
#  STEP 9 — XGBoost
# =============================================================================
print('\n' + '='*68)
print('  STEP 9 — XGBoost')
print('='*68)

XGB_PATH = os.path.join(OUT_DIR, 'xgb_model.pkl')
if os.path.exists(XGB_PATH):
    print('Loading saved XGBoost...')
    xgb_m = joblib.load(XGB_PATH)
else:
    print('Training...')
    t0 = time.time()
    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42, n_jobs=-1, tree_method='hist'
    )
    xgb_m.fit(X_tr_sc, y_tr,
              eval_set=[(X_te_sc, y_te)],
              verbose=50)
    joblib.dump(xgb_m, XGB_PATH)
    print(f'Done in {time.time()-t0:.1f}s')

xgb_pred  = xgb_m.predict(X_te_sc)
xgb_proba = xgb_m.predict_proba(X_te_sc)[:, 1]
print('\n--- XGBoost ---')
evaluate('XGBoost', y_te, xgb_pred, xgb_proba)
print(classification_report(y_te, xgb_pred, target_names=['Normal','Attack']))

# =============================================================================
#  STEP 10 — MLP
# =============================================================================
print('\n' + '='*68)
print('  STEP 10 — MLP Neural Network')
print('='*68)

MLP_PATH = os.path.join(OUT_DIR, 'mlp_model.pkl')
if os.path.exists(MLP_PATH):
    print('Loading saved MLP...')
    mlp = joblib.load(MLP_PATH)
else:
    print('Training...')
    t0 = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu', solver='adam',
        alpha=0.001, learning_rate_init=1e-3,
        max_iter=200, early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15, random_state=42
    )
    mlp.fit(X_tr_sc, y_tr)
    joblib.dump(mlp, MLP_PATH)
    print(f'Stopped at iteration {mlp.n_iter_}  ({time.time()-t0:.1f}s)')

mlp_pred  = mlp.predict(X_te_sc)
mlp_proba = mlp.predict_proba(X_te_sc)[:, 1]
print('\n--- MLP Neural Network ---')
evaluate('MLP Neural Network', y_te, mlp_pred, mlp_proba)
print(classification_report(y_te, mlp_pred, target_names=['Normal','Attack']))

# =============================================================================
#  STEP 11 — Results Table
# =============================================================================
print('\n' + '='*68)
print('  MODEL COMPARISON — CAN Bus Dataset (70/30 stratified split)')
print('='*68)

res_df = pd.DataFrame(all_results).T
res_df.to_csv(os.path.join(OUT_DIR, 'model_results.csv'))
print(res_df.to_string())
print('='*68)
print(f'\nBest: {res_df["Accuracy"].idxmax()}  ->  {res_df["Accuracy"].max():.2f}%')
for m, row in res_df.iterrows():
    acc = row['Accuracy']
    tag = '[IN TARGET]' if 88<=acc<=92 else ('[ABOVE]' if acc>92 else '[BELOW]')
    print(f'  {m:<25s}: {acc:.2f}%  {tag}')

# =============================================================================
#  STEP 12 — Plots
# =============================================================================
print('\n' + '='*68)
print('  STEP 12 — Generating Plots')
print('='*68)

colors_m = ['#3498db','#2ecc71','#e74c3c','#9b59b6']
mlist    = list(all_results.keys())
mkeys    = ['Accuracy','Precision','Recall','F1','ROC-AUC']

# Model comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Comparison - CAN Bus Dataset', fontsize=14, fontweight='bold')
x = np.arange(len(mkeys)); w = 0.18
for idx, (mn, col) in enumerate(zip(mlist, colors_m)):
    vals = [all_results[mn][k] for k in mkeys]
    axes[0].bar(x+idx*w, vals, w, label=mn, color=col, alpha=0.85)
axes[0].set_xticks(x+w*1.5); axes[0].set_xticklabels(mkeys, fontsize=9)
axes[0].set_ylim(0, 112); axes[0].set_ylabel('Score (%)')
axes[0].set_title('All Metrics'); axes[0].legend(fontsize=7)
axes[0].axhline(88, color='gray', ls='--', lw=1)
axes[0].axhline(92, color='gray', ls=':', lw=1)
axes[0].grid(axis='y', alpha=0.3)

roc_data = [('LR',lr_proba),('RF',rf_proba),('XGB',xgb_proba),('MLP',mlp_proba)]
for (mn, yp), col in zip(roc_data, colors_m):
    fpr, tpr, _ = roc_curve(y_te, yp)
    auc = roc_auc_score(y_te, yp)
    axes[1].plot(fpr, tpr, color=col, lw=2, label=f'{mn} ({auc*100:.1f}%)')
axes[1].plot([0,1],[0,1],'k--',lw=1)
axes[1].set_title('ROC Curves')
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

accs = {m: v['Accuracy'] for m, v in all_results.items()}
bars = axes[2].bar(list(accs.keys()), list(accs.values()), color=colors_m, alpha=0.85)
axes[2].axhspan(88, 92, color='yellow', alpha=0.25, label='Target 88-92%')
axes[2].set_ylim(50, 100); axes[2].set_ylabel('Accuracy (%)')
axes[2].set_title('Accuracy vs Target')
axes[2].legend(fontsize=8); axes[2].tick_params(axis='x', rotation=15)
for bar, val in zip(bars, accs.values()):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)
axes[2].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Saved: model_comparison.png')

# Confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
fig.suptitle('Confusion Matrices - CAN Bus Dataset', fontsize=13, fontweight='bold')
for ax, (name, yt, yp) in zip(axes, [
    ('LR',  y_te, lr_pred), ('RF',  y_te, rf_pred),
    ('XGB', y_te, xgb_pred), ('MLP', y_te, mlp_pred),
]):
    cm  = confusion_matrix(yt, yp)
    pct = cm.astype(float)/cm.sum()*100
    ann = np.array([[f'{v}\n({p:.1f}%)' for v,p in zip(rc,rp)]
                    for rc,rp in zip(cm,pct)])
    sns.heatmap(cm, annot=ann, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Normal','Attack'],
                yticklabels=['Normal','Attack'], annot_kws={'size':9})
    ax.set_title(f'{name}\nacc={accuracy_score(yt,yp)*100:.1f}%', fontsize=10)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Saved: confusion_matrices.png')

# XGBoost feature importance
fi = pd.Series(xgb_m.feature_importances_, index=FEAT_COLS).sort_values()
plt.figure(figsize=(10, 8))
fi.plot(kind='barh',
        color=['#e74c3c' if v>fi.quantile(0.75) else '#3498db' for v in fi])
plt.title('XGBoost Feature Importance', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.axvline(fi.quantile(0.75), color='black', ls='--', lw=1, label='Top 25%')
plt.legend(fontsize=9); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Saved: feature_importance.png')
print('Top features:')
print(fi.tail(10).round(4).to_string())

# Per-attack-type breakdown
type_rows = []
for atype in np.unique(g_te):
    mask = g_te == atype
    if mask.sum() < 20: continue
    yt, yp, ypr = y_te[mask], xgb_pred[mask], xgb_proba[mask]
    if len(np.unique(yt)) < 2: continue
    type_rows.append({
        'Attack Type': atype, 'N': int(mask.sum()),
        'Accuracy'   : round(accuracy_score(yt,yp)*100, 1),
        'Precision'  : round(precision_score(yt,yp,zero_division=0)*100, 1),
        'Recall'     : round(recall_score(yt,yp,zero_division=0)*100, 1),
        'F1'         : round(f1_score(yt,yp,zero_division=0)*100, 1),
        'ROC-AUC'    : round(roc_auc_score(yt,ypr)*100, 1),
    })
type_df = pd.DataFrame(type_rows)
type_df.to_csv(os.path.join(OUT_DIR,'per_attack_type_results.csv'), index=False)
print('\nXGBoost - Per Attack Type:')
print(type_df.to_string(index=False))

if len(type_df) > 0:
    fig, ax = plt.subplots(figsize=(14, 5))
    x2 = np.arange(len(type_df)); w2 = 0.2
    for i, (cn, cc) in enumerate(zip(['Accuracy','Precision','Recall','F1'],
                                      ['#3498db','#2ecc71','#e74c3c','#9b59b6'])):
        ax.bar(x2+i*w2, type_df[cn], w2, label=cn, color=cc, alpha=0.85)
    ax.set_xticks(x2+w2*1.5)
    ax.set_xticklabels(type_df['Attack Type'], rotation=25, ha='right', fontsize=9)
    ax.set_ylim(0, 112); ax.set_ylabel('Score (%)')
    ax.set_title('XGBoost - Per Attack Type Performance', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'per_attack_type.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: per_attack_type.png')

# =============================================================================
#  STEP 13 — Final Summary
# =============================================================================
print('\n' + '='*68)
print('  FINAL SUMMARY')
print('='*68)
print(f'  Features     : {len(FEAT_COLS)}')
print(f'  Train samples: {len(X_tr):,}')
print(f'  Test samples : {len(X_te):,}')
print()
print(f'  {"Model":<25s}  {"Accuracy":>10s}  {"F1":>8s}  {"ROC-AUC":>10s}  Status')
print(f'  {"-"*25}  {"-"*10}  {"-"*8}  {"-"*10}  ------')
for m, r in res_df.iterrows():
    acc = r['Accuracy']
    tag = '[TARGET]' if 88<=acc<=92 else ('[ABOVE]' if acc>92 else '[BELOW]')
    print(f'  {m:<25s}  {acc:>9.2f}%  {r["F1"]:>7.2f}%  {r["ROC-AUC"]:>9.2f}%  {tag}')
print()
print(f'  Output: {OUT_DIR}')
files = ['scaler.pkl','lr_model.pkl','rf_model.pkl','xgb_model.pkl','mlp_model.pkl',
         'model_comparison.png','confusion_matrices.png','feature_importance.png',
         'per_attack_type.png','model_results.csv','per_attack_type_results.csv']
for fname in files:
    path = os.path.join(OUT_DIR, fname)
    ok   = os.path.exists(path)
    sz   = f'({os.path.getsize(path)/1e6:.1f}MB)' if ok else ''
    print(f'  {"OK" if ok else "MISSING"} {fname} {sz}')
print('='*68)
ram()