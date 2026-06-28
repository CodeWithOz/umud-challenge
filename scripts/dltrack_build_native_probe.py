"""Build a static probe from DL_Track NATIVE geometry (its own doCalculations).
Calibrate to LB centers keeping most per-image variation; block19 for invalid rows."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path('/Users/ucheozoemena/01-projects/umud-challenge')
dl = pd.read_csv(ROOT / 'data/dltrack-models/dltrack_native_geom.csv')
b19 = pd.read_csv(ROOT / 'data/kaggle-outputs/block19-cxs5-refresh/submit/submission.csv').sort_values('image_id').reset_index(drop=True)
PX = 0.0881
if 'fl_mm' not in dl.columns:
    dl['fl_mm'] = dl['fl_px'] * PX
    dl['mt_mm'] = dl['mt_px'] * PX

d = dl.dropna(subset=['pa_deg', 'fl_mm', 'mt_mm']).copy()
d['mt_mm'] = d['mt_mm'].clip(10, 32)   # kill apo-detection MT outliers (raw max 881px)
# Calibrate to LB centers, keep most per-image variation (DL_Track is documented-accurate).
S = {'pa': 0.8, 'fl': 0.8, 'mt': 0.6}
C = {'pa': 16.5, 'fl': 76.0, 'mt': 20.0}
cal = pd.DataFrame({'image_id': d['image_id']})
cal['pa_deg'] = (C['pa'] + S['pa'] * (d['pa_deg'] - d['pa_deg'].median())).clip(1, 45)
cal['fl_mm'] = (C['fl'] + S['fl'] * (d['fl_mm'] - d['fl_mm'].median())).clip(30, 200)
cal['mt_mm'] = (C['mt'] + S['mt'] * (d['mt_mm'] - d['mt_mm'].median())).clip(8, 50)

cal_map = cal.set_index('image_id'); b19_map = b19.set_index('image_id')
final = b19[['image_id']].copy()
for c in ['pa_deg', 'fl_mm', 'mt_mm']:
    final[c] = final['image_id'].map(cal_map[c]).fillna(final['image_id'].map(b19_map[c]))
final = final[['image_id', 'pa_deg', 'fl_mm', 'mt_mm']]
assert len(final) == 309 and final[['pa_deg', 'fl_mm', 'mt_mm']].notna().all().all()
out = ROOT / 'data/dltrack-models/submission_dltrack_native_probe.csv'
final.to_csv(out, index=False)
print(f"DL_Track native rows: {len(cal)}/309 ({len(cal)/309*100:.0f}%), rest block19")
print("raw native medians: pa %.1f fl %.1f mt %.1f" % (d.pa_deg.median(), d.fl_mm.median(), d.mt_mm.median()))
print("calibrated dist:")
print(cal[['pa_deg', 'fl_mm', 'mt_mm']].describe(percentiles=[.5]).round(2).loc[['mean', 'std', '50%']])
print("wrote", out)
