import os, gc, time, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from tqdm import tqdm
import psutil
from sklearn.metrics import (confusion_matrix, classification_report,
                              precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve)

warnings.filterwarnings('ignore')

# =============================================================================
OTIDS_OUT = '/Users/tarun/Desktop/otids_output'
RES_DIR   = '/Users/tarun/Desktop/nb5_dashboard'
# =============================================================================

os.makedirs(RES_DIR, exist_ok=True)

FEAT_COLS = [
    'byte_0','byte_1','byte_2','byte_3','byte_4','byte_5','byte_6','byte_7',
    'byte_mean','byte_std','byte_min','byte_max','byte_range','byte_entropy',
    'iat','iat_zscore','freq_dev','hamming','dlc_change',
    'roll_iat_mean','roll_iat_std','roll_entropy_mean','arb_id_enc'
]

ALERT_THRESHOLD = 0.5
MAX_MESSAGES    = 10_000

def ram():
    p = psutil.virtual_memory()
    print(f'  RAM  used={(p.total-p.available)/1e9:.1f}GB  avail={p.available/1e9:.1f}GB')

# =============================================================================
#  STEP 1 — Load models and data
# =============================================================================
print('='*68)
print('  STEP 1 — Loading models and data')
print('='*68)

for f in ['xgb_model.pkl','rf_model.pkl','mlp_model.pkl',
          'scaler.pkl','otids_features.parquet']:
    if not os.path.exists(os.path.join(OTIDS_OUT, f)):
        raise FileNotFoundError(f'Missing: {os.path.join(OTIDS_OUT, f)}\nRun nb2_otids.py first.')

xgb_model = joblib.load(os.path.join(OTIDS_OUT, 'xgb_model.pkl'))
rf_model  = joblib.load(os.path.join(OTIDS_OUT, 'rf_model.pkl'))
mlp_model = joblib.load(os.path.join(OTIDS_OUT, 'mlp_model.pkl'))
scaler    = joblib.load(os.path.join(OTIDS_OUT, 'scaler.pkl'))
print('Loaded: XGBoost, RF, MLP, Scaler')

df = pd.read_parquet(os.path.join(OTIDS_OUT, 'otids_features.parquet'))
for col in FEAT_COLS:
    if col not in df.columns:
        df[col] = 0.0

keep = FEAT_COLS + ['label','timestamp','arb_id','attack_type']
keep = [c for c in keep if c in df.columns]
df   = df[keep].dropna()
df[FEAT_COLS] = df[FEAT_COLS].fillna(0).replace([np.inf, -np.inf], 0)

print(f'Full dataset: {len(df):,} rows | attack={df["label"].mean()*100:.2f}%')
print(df['attack_type'].value_counts().to_string() if 'attack_type' in df.columns else '')

# =============================================================================
#  STEP 2 — BALANCED held-out stream
# =============================================================================
print('\n' + '='*68)
print('  STEP 2 — Building balanced simulation stream')
print('='*68)

normal_df = df[df['label'] == 0]
attack_df = df[df['label'] == 1]

print(f'Normal rows: {len(normal_df):,}  Attack rows: {len(attack_df):,}')

# Take 20% held-out from each class
n_normal_stream = min(int(len(normal_df) * 0.2), MAX_MESSAGES // 2)
n_attack_stream = min(int(len(attack_df) * 0.2), MAX_MESSAGES // 2)

# Make sure we have enough attacks
n_attack_stream = max(n_attack_stream, min(len(attack_df), 500))
n_normal_stream = min(n_normal_stream, MAX_MESSAGES - n_attack_stream)

stream_normal = normal_df.sample(n=n_normal_stream, random_state=99)
stream_attack = attack_df.sample(n=n_attack_stream, random_state=99)

# Interleave: sort both by timestamp to simulate realistic message ordering
stream = pd.concat([stream_normal, stream_attack])
if 'timestamp' in stream.columns:
    stream = stream.sort_values('timestamp').reset_index(drop=True)
else:
    stream = stream.sample(frac=1, random_state=42).reset_index(drop=True)
    stream['timestamp'] = np.arange(len(stream)) * 0.001

del normal_df, attack_df; gc.collect()

print(f'Stream: {len(stream):,} messages | attack={stream["label"].mean()*100:.2f}%')
if 'attack_type' in stream.columns:
    print(stream['attack_type'].value_counts().to_string())
ram()

# =============================================================================
#  STEP 3 — Run Real-Time Simulation 
# =============================================================================
print('\n' + '='*68)
print('  STEP 3 — Running Real-Time Simulation')
print('='*68)

X_all  = scaler.transform(stream[FEAT_COLS].values.astype(np.float32))
y_all  = stream['label'].values.astype(np.int32)
ts_all = stream['timestamp'].values
id_all = stream['arb_id'].values if 'arb_id' in stream.columns else ['?']*len(stream)
at_all = stream['attack_type'].values if 'attack_type' in stream.columns else ['Unknown']*len(stream)
del stream; gc.collect()

results   = []
latencies = []

print(f'Simulating {len(X_all):,} messages (threshold={ALERT_THRESHOLD})...')
for i in tqdm(range(len(X_all))):
    x     = X_all[i:i+1]
    t0    = time.perf_counter()
    score = float(xgb_model.predict_proba(x)[0][1])
    lat   = (time.perf_counter() - t0) * 1000

    latencies.append(lat)
    alert = score > ALERT_THRESHOLD
    tl    = int(y_all[i])

    results.append({
        'idx'        : i,
        'timestamp'  : float(ts_all[i]),
        'arb_id'     : str(id_all[i]),
        'attack_type': str(at_all[i]),
        'score'      : score,
        'alert'      : int(alert),
        'true_label' : tl,
        'tp'         : int(alert and tl==1),
        'fp'         : int(alert and tl==0),
        'tn'         : int(not alert and tl==0),
        'fn'         : int(not alert and tl==1),
        'latency_ms' : lat
    })

res = pd.DataFrame(results)
lat = np.array(latencies)

tp = res['tp'].sum(); fp = res['fp'].sum()
tn = res['tn'].sum(); fn = res['fn'].sum()
prec = tp / max(tp+fp, 1)
rec  = tp / max(tp+fn, 1)
f1   = 2*prec*rec / max(prec+rec, 1e-9)
auc  = roc_auc_score(res['true_label'], res['score']) if res['true_label'].nunique()>1 else 0.0

print(f'\n=== Simulation Results (threshold={ALERT_THRESHOLD}) ===')
print(f'Messages : {len(res):,}  |  Alerts: {res["alert"].sum():,}')
print(f'TP={tp}  FP={fp}  TN={tn}  FN={fn}')
print(f'Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}')
print(f'Latency  mean={lat.mean():.3f}ms  p99={np.percentile(lat,99):.3f}ms')
ram()

# =============================================================================
#  STEP 4 — Multi-Model Latency Benchmark
# =============================================================================
print('\n' + '='*68)
print('  STEP 4 — Multi-Model Latency Benchmark')
print('='*68)

BENCH_N = min(1000, len(X_all))
bench_results = {}
for name, model in [('XGBoost',xgb_model),('RF',rf_model),('MLP',mlp_model)]:
    lats = []
    for i in range(BENCH_N):
        t0 = time.perf_counter()
        _  = model.predict_proba(X_all[i:i+1])
        lats.append((time.perf_counter()-t0)*1000)
    lats = np.array(lats)
    bench_results[name] = {
        'mean'  : round(lats.mean(),   4),
        'median': round(np.median(lats),4),
        'p95'   : round(np.percentile(lats,95),4),
        'p99'   : round(np.percentile(lats,99),4),
        'std'   : round(lats.std(),    4),
    }
    print(f'  {name:<12s} mean={lats.mean():.3f}ms  '
          f'median={np.median(lats):.3f}ms  p99={np.percentile(lats,99):.3f}ms')

bench_df = pd.DataFrame(bench_results).T
bench_df.to_csv(os.path.join(RES_DIR,'latency_benchmark.csv'))

# =============================================================================
#  STEP 5 — Threshold Optimisation
# =============================================================================
print('\n' + '='*68)
print('  STEP 5 — Threshold Optimisation')
print('='*68)

thresholds = np.arange(0.05, 0.96, 0.05)
thr_rows   = []
for t in thresholds:
    preds = (res['score'] > t).astype(int)
    thr_rows.append({
        'threshold': round(float(t), 2),
        'precision': precision_score(res['true_label'], preds, zero_division=0),
        'recall'   : recall_score(res['true_label'],    preds, zero_division=0),
        'f1'       : f1_score(res['true_label'],        preds, zero_division=0),
        'fpr'      : ((preds==1)&(res['true_label']==0)).sum() /
                     max((res['true_label']==0).sum(), 1)
    })

tdf    = pd.DataFrame(thr_rows)
best_t = float(tdf.loc[tdf['f1'].idxmax(), 'threshold'])
best_r = tdf.loc[tdf['f1'].idxmax()]
print(f'Optimal threshold : {best_t}')
print(f'  F1={best_r["f1"]:.4f}  Recall={best_r["recall"]:.4f}  '
      f'Precision={best_r["precision"]:.4f}  FPR={best_r["fpr"]:.4f}')

final_preds = (res['score'] > best_t).astype(int)
opt_f1   = f1_score(res['true_label'],        final_preds, zero_division=0)
opt_rec  = recall_score(res['true_label'],    final_preds, zero_division=0)
opt_prec = precision_score(res['true_label'], final_preds, zero_division=0)
opt_auc  = roc_auc_score(res['true_label'], res['score']) if res['true_label'].nunique()>1 else 0
cm_opt   = confusion_matrix(res['true_label'], final_preds)

# =============================================================================
#  STEP 6 — 9-Panel Dashboard
# =============================================================================
print('\n' + '='*68)
print('  STEP 6 — Generating 9-Panel Dashboard')
print('='*68)

W = 100
res['r_tpr'] = (res['tp'].rolling(W).sum() /
                (res['tp'].rolling(W).sum() + res['fn'].rolling(W).sum() + 1e-9))
res['r_fpr'] = (res['fp'].rolling(W).sum() /
                (res['fp'].rolling(W).sum() + res['tn'].rolling(W).sum() + 1e-9))

fig = plt.figure(figsize=(20, 14))
fig.suptitle('CAN Bus IDS — Real-Time Monitoring Dashboard',
             fontsize=16, fontweight='bold', y=0.98)

# Panel 1: Threat score over time (full top row)
ax1 = fig.add_subplot(3, 3, (1, 3))
s   = res.tail(1000).reset_index(drop=True)
norm_mask = s['true_label'] == 0
atk_mask  = s['true_label'] == 1
ax1.fill_between(s.index, s['score'], where=norm_mask,
                  alpha=0.5, color='#2ecc71', label='Normal')
ax1.fill_between(s.index, s['score'], where=atk_mask,
                  alpha=0.7, color='#e74c3c', label='Attack')
ax1.axhline(ALERT_THRESHOLD, color='#f39c12', ls='--', lw=2,
             label=f'Threshold ({ALERT_THRESHOLD})')
ax1.axhline(best_t, color='#8e44ad', ls=':', lw=1.5,
             label=f'Optimal ({best_t})')
fp_idx = s[s['fp']==1].index
if len(fp_idx):
    ax1.scatter(fp_idx, s.loc[fp_idx,'score'], color='yellow',
                s=25, zorder=5, label='False Positives', alpha=0.8)
ax1.set_title('Threat Score — Last 1000 Messages', fontweight='bold')
ax1.set_ylabel('Attack Probability')
ax1.set_ylim(-0.05, 1.1)
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(alpha=0.3)

# Panel 2: Rolling TPR / FPR
ax2 = fig.add_subplot(3, 3, 4)
ax2.plot(res.index, res['r_tpr'], color='#2ecc71', lw=1.5, label='TPR (Recall)')
ax2.plot(res.index, res['r_fpr'], color='#e74c3c', lw=1.5, label='FPR')
ax2.set_title(f'Rolling TPR & FPR (window={W})', fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.legend(fontsize=8)
ax2.set_xlabel('Message index')
ax2.grid(alpha=0.3)

# Panel 3: Score distribution
ax3 = fig.add_subplot(3, 3, 5)
norm_scores = res[res['true_label']==0]['score']
atk_scores  = res[res['true_label']==1]['score']
if len(norm_scores): ax3.hist(norm_scores, bins=50, alpha=0.65,
                               color='#2ecc71', label=f'Normal (n={len(norm_scores):,})', density=True)
if len(atk_scores):  ax3.hist(atk_scores,  bins=50, alpha=0.65,
                               color='#e74c3c', label=f'Attack (n={len(atk_scores):,})',  density=True)
ax3.axvline(ALERT_THRESHOLD, color='#f39c12', ls='--', lw=2, label=f'Default={ALERT_THRESHOLD}')
ax3.axvline(best_t,          color='#8e44ad', ls=':',  lw=1.5, label=f'Optimal={best_t}')
ax3.set_title('Score Distribution', fontweight='bold')
ax3.set_xlabel('Attack Probability')
ax3.legend(fontsize=7)
ax3.grid(alpha=0.3)

# Panel 4: Confusion matrix at optimal threshold
ax4 = fig.add_subplot(3, 3, 6)
pct = cm_opt.astype(float) / cm_opt.sum() * 100
ann = np.array([[f'{v:,}\n({p:.1f}%)' for v,p in zip(rc,rp)]
                for rc,rp in zip(cm_opt, pct)])
sns.heatmap(cm_opt, annot=ann, fmt='', cmap='Blues', ax=ax4,
            xticklabels=['Normal','Attack'],
            yticklabels=['Normal','Attack'],
            annot_kws={'size':11, 'weight':'bold'})
ax4.set_title(f'Confusion Matrix (thr={best_t})\n'
              f'F1={opt_f1:.3f}  Recall={opt_rec:.3f}  Precision={opt_prec:.3f}',
              fontweight='bold')
ax4.set_xlabel('Predicted'); ax4.set_ylabel('True')

# Panel 5: Latency histogram
ax5 = fig.add_subplot(3, 3, 7)
ax5.hist(lat[lat < np.percentile(lat,99.5)], bins=60,
          color='#3498db', alpha=0.8, edgecolor='white', lw=0.3)
ax5.axvline(lat.mean(), color='#e74c3c', lw=2, ls='--',
             label=f'Mean={lat.mean():.3f}ms')
ax5.axvline(np.percentile(lat,99), color='#f39c12', lw=2, ls=':',
             label=f'P99={np.percentile(lat,99):.3f}ms')
ax5.set_title('XGBoost Inference Latency', fontweight='bold')
ax5.set_xlabel('Latency (ms)')
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

# Panel 6: Detection rate by attack type
ax6 = fig.add_subplot(3, 3, 8)
atk_only = res[res['true_label']==1]
if len(atk_only) > 0 and 'attack_type' in atk_only.columns:
    tdr = atk_only.groupby('attack_type').apply(
        lambda x: pd.Series({
            'TPR'  : x['tp'].sum() / max(len(x), 1),
            'Count': len(x)
        })
    ).sort_values('TPR')
    bar_colors = ['#e74c3c' if v<0.6 else '#f39c12' if v<0.85 else '#2ecc71'
                  for v in tdr['TPR']]
    ax6.barh(tdr.index, tdr['TPR'], color=bar_colors)
    ax6.set_title('Detection Rate by Attack Type', fontweight='bold')
    ax6.set_xlabel('True Positive Rate')
    ax6.set_xlim(0, 1.15)
    ax6.grid(alpha=0.3)
    for i,(v,c) in enumerate(zip(tdr['TPR'], tdr['Count'])):
        ax6.text(v+0.01, i, f'{v:.3f} (n={int(c)})', va='center', fontsize=8)
else:
    ax6.text(0.5, 0.5, 'No attack type data', ha='center', va='center',
             transform=ax6.transAxes, fontsize=12)
    ax6.set_title('Detection Rate by Attack Type', fontweight='bold')

# Panel 7: Threshold tuning
ax7 = fig.add_subplot(3, 3, 9)
ax7.plot(tdf['threshold'], tdf['precision'], 'b-o', ms=3, lw=1.5, label='Precision')
ax7.plot(tdf['threshold'], tdf['recall'],    'r-o', ms=3, lw=1.5, label='Recall')
ax7.plot(tdf['threshold'], tdf['f1'],        'g-o', ms=4, lw=2.5, label='F1')
ax7.plot(tdf['threshold'], tdf['fpr'],       'm--o', ms=3, lw=1.5, label='FPR')
ax7.axvline(best_t, color='black', ls=':', lw=2, label=f'Best={best_t}')
ax7.set_title('Threshold Tuning', fontweight='bold')
ax7.set_xlabel('Alert Threshold')
ax7.set_ylim(0, 1.1)
ax7.legend(fontsize=7)
ax7.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RES_DIR,'realtime_dashboard.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Saved: realtime_dashboard.png')

# =============================================================================
#  STEP 7 — Summary card
# =============================================================================
fig2, ax_s = plt.subplots(figsize=(8, 7))
ax_s.axis('off')
tp2 = cm_opt[1,1] if cm_opt.shape==(2,2) else 0
fp2 = cm_opt[0,1] if cm_opt.shape==(2,2) else 0
tn2 = cm_opt[0,0] if cm_opt.shape==(2,2) else 0
fn2 = cm_opt[1,0] if cm_opt.shape==(2,2) else 0

txt = (
    f"CAN BUS IDS — SIMULATION SUMMARY\n"
    f"{'='*38}\n"
    f"Dataset          : OTIDS (CAN-Intrusion)\n"
    f"Messages streamed: {len(res):,}\n"
    f"  Normal         : {(res['true_label']==0).sum():,}\n"
    f"  Attack         : {(res['true_label']==1).sum():,}\n"
    f"Alerts raised    : {res['alert'].sum():,}\n"
    f"{'─'*38}\n"
    f"Default thr ({ALERT_THRESHOLD}):\n"
    f"  Precision      : {prec:.4f}\n"
    f"  Recall         : {rec:.4f}\n"
    f"  F1 Score       : {f1:.4f}\n"
    f"  ROC-AUC        : {auc:.4f}\n"
    f"{'─'*38}\n"
    f"Optimal thr ({best_t}):\n"
    f"  Precision      : {opt_prec:.4f}\n"
    f"  Recall         : {opt_rec:.4f}\n"
    f"  F1 Score       : {opt_f1:.4f}\n"
    f"  ROC-AUC        : {opt_auc:.4f}\n"
    f"  TP={tp2}  FP={fp2}  TN={tn2}  FN={fn2}\n"
    f"{'─'*38}\n"
    f"Latency (XGBoost per message):\n"
    f"  Mean           : {lat.mean():.4f} ms\n"
    f"  Median         : {np.median(lat):.4f} ms\n"
    f"  Std            : {lat.std():.4f} ms\n"
    f"  P95            : {np.percentile(lat,95):.4f} ms\n"
    f"  P99            : {np.percentile(lat,99):.4f} ms\n"
)
ax_s.text(0.05, 0.97, txt, transform=ax_s.transAxes, fontsize=10,
           va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#f0f4f8', alpha=0.9))
fig2.suptitle('Simulation Summary Card', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR,'summary_card.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Saved: summary_card.png')

# Threshold tuning standalone
plt.figure(figsize=(9, 5))
plt.plot(tdf['threshold'], tdf['precision'], 'b-o', ms=4, label='Precision')
plt.plot(tdf['threshold'], tdf['recall'],    'r-o', ms=4, label='Recall')
plt.plot(tdf['threshold'], tdf['f1'],        'g-o', ms=4, lw=2.5, label='F1')
plt.plot(tdf['threshold'], tdf['fpr'],       'm--o', ms=4, label='FPR')
plt.axvline(best_t, color='black', ls=':', lw=2, label=f'Best={best_t}')
plt.title('Threshold Tuning — CAN Bus IDS', fontweight='bold')
plt.xlabel('Alert Threshold'); plt.ylabel('Score')
plt.ylim(0, 1.1); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR,'threshold_tuning.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Saved: threshold_tuning.png')

# Multi-model latency comparison bar chart
fig3, ax_lat = plt.subplots(figsize=(8, 4))
models_lat = list(bench_results.keys())
means_lat  = [bench_results[m]['mean']   for m in models_lat]
p99_lat    = [bench_results[m]['p99']    for m in models_lat]
x_lat = np.arange(len(models_lat)); w_lat = 0.35
ax_lat.bar(x_lat-w_lat/2, means_lat, w_lat, label='Mean',  color='#3498db', alpha=0.85)
ax_lat.bar(x_lat+w_lat/2, p99_lat,   w_lat, label='P99',   color='#e74c3c', alpha=0.85)
ax_lat.set_xticks(x_lat); ax_lat.set_xticklabels(models_lat)
ax_lat.set_ylabel('Latency (ms)'); ax_lat.set_title('Model Inference Latency Comparison', fontweight='bold')
ax_lat.legend(); ax_lat.grid(axis='y', alpha=0.3)
for i,(m,p) in enumerate(zip(means_lat, p99_lat)):
    ax_lat.text(i-w_lat/2, m+0.002, f'{m:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax_lat.text(i+w_lat/2, p+0.002, f'{p:.3f}', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR,'latency_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Saved: latency_comparison.png')

# =============================================================================
#  STEP 8 — Final classification report
# =============================================================================
print('\n' + '='*68)
print('  FINAL CLASSIFICATION REPORT')
print('='*68)
print(f'\n--- Using optimal threshold = {best_t} ---')
print(classification_report(res['true_label'], final_preds,
                              target_names=['Normal','Attack']))
print('=== INFERENCE LATENCY ===')
for label, val in [
    ('Mean  ', lat.mean()),   ('Median', np.median(lat)),
    ('Std   ', lat.std()),    ('P95   ', np.percentile(lat,95)),
    ('P99   ', np.percentile(lat,99)),
    ('Min   ', lat.min()),    ('Max   ', lat.max()),
]:
    print(f'  {label}: {val:.4f} ms')

print('\n=== MULTI-MODEL LATENCY ===')
print(bench_df.to_string())

# Save CSVs
res.to_csv(os.path.join(RES_DIR,'simulation_results.csv'), index=False)
tdf.to_csv(os.path.join(RES_DIR,'threshold_tuning.csv'),   index=False)

# =============================================================================
#  FINAL SUMMARY
# =============================================================================
print('\n' + '='*68)
print('  NB5 COMPLETE')
print('='*68)
print(f'  Messages simulated : {len(res):,}')
print(f'    Normal           : {(res["true_label"]==0).sum():,}')
print(f'    Attack           : {(res["true_label"]==1).sum():,}')
print(f'  Optimal threshold  : {best_t}')
print(f'  F1 (optimal)       : {opt_f1:.4f}')
print(f'  Recall (optimal)   : {opt_rec:.4f}')
print(f'  Precision (optimal): {opt_prec:.4f}')
print(f'  ROC-AUC            : {opt_auc:.4f}')
print(f'  Latency mean       : {lat.mean():.4f} ms')
print(f'  Latency P99        : {np.percentile(lat,99):.4f} ms')
print(f'\n  Output saved to: {RES_DIR}')
files = ['realtime_dashboard.png','summary_card.png','threshold_tuning.png',
         'latency_comparison.png','simulation_results.csv',
         'threshold_tuning.csv','latency_benchmark.csv']
for fname in files:
    path = os.path.join(RES_DIR, fname)
    ok   = os.path.exists(path)
    sz   = f'({os.path.getsize(path)/1e6:.2f}MB)' if ok else ''
    print(f'  {"OK" if ok else "MISSING"} {fname} {sz}')
print('='*68)
ram()
print('\nProject complete!')