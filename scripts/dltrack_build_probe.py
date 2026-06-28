"""Build a static-CSV probe from DL_Track local geometry: calibrate to LB centers,
use block19 for the rows we don't have locally. Public diagnostic (no GPU, no Kaggle run)."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path('/Users/ucheozoemena/01-projects/umud-challenge')
dl = pd.read_csv(ROOT / 'data/dltrack-models/dltrack_raw_geom.csv')
b19 = pd.read_csv(ROOT / 'data/kaggle-outputs/block19-cxs5-refresh/submit/submission.csv').sort_values('image_id').reset_index(drop=True)

# Calibrate DL_Track raw -> LB centers, moderate shrink (keep real per-image variation).
d = dl.dropna(subset=['pa_raw', 'mt_raw_mm', 'fl_raw_mm']).copy()
flc = d['fl_raw_mm'].clip(30, 250)
cal = pd.DataFrame({'image_id': d['image_id']})
cal['pa_deg'] = (16.5 + 0.7 * (d['pa_raw'] - d['pa_raw'].median())).clip(1, 45)
cal['mt_mm'] = (20.0 + 0.5 * (d['mt_raw_mm'] - d['mt_raw_mm'].median())).clip(8, 50)
cal['fl_mm'] = (76.0 + 0.4 * (flc - flc.median())).clip(30, 200)

# 309-row submission: DL_Track-calibrated where available, else block19.
final = b19[['image_id']].copy()
cal_map = cal.set_index('image_id')
b19_map = b19.set_index('image_id')
for c in ['pa_deg', 'fl_mm', 'mt_mm']:
    dlv = final['image_id'].map(cal_map[c])
    b19v = final['image_id'].map(b19_map[c])
    final[c] = dlv.fillna(b19v)
final = final[['image_id', 'pa_deg', 'fl_mm', 'mt_mm']]
assert len(final) == 309 and final[['pa_deg', 'fl_mm', 'mt_mm']].notna().all().all()
out = ROOT / 'data/dltrack-models/submission_dltrack_probe.csv'
final.to_csv(out, index=False)
print('DL_Track rows used:', cal_map.shape[0], '/ 309 (rest block19)')
print('calibrated DL_Track dist:')
print(cal[['pa_deg', 'fl_mm', 'mt_mm']].describe(percentiles=[.5]).round(2).loc[['mean', 'std', '50%']])
print('final 309 dist:')
print(final[['pa_deg', 'fl_mm', 'mt_mm']].describe(percentiles=[.5]).round(2).loc[['mean', 'std', '50%']])
print('wrote', out)
