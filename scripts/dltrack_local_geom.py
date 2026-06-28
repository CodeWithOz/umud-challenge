"""Run DL_Track pretrained VGG16 apo+fascicle models locally on the test images and
compute raw PA/FL/MT geometry. No Kaggle GPU; CPU TensorFlow via tf_keras (Keras 2)."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import math
from pathlib import Path
import numpy as np, cv2, pandas as pd
import scipy.ndimage as ndi
import tf_keras as keras

ROOT = Path('/Users/ucheozoemena/01-projects/umud-challenge')
fasc = keras.models.load_model(ROOT / 'data/dltrack-models/model-fasc-VGG16.h5', compile=False)
apo = keras.models.load_model(ROOT / 'data/dltrack-models/model-apo-VGG16.h5', compile=False)
PIXEL_TO_MM = 0.0881


def predict(m, img):
    r = cv2.resize(img, (512, 512))
    rgb = np.stack([r, r, r], -1).astype(np.float32) / 255.0
    return m.predict(rgb[None], verbose=0)[0, ..., 0]


def deep_sup(ab, oh):
    ys, xs = np.where(ab > 0)
    if len(ys) < 50:
        return None
    mid = (ys.min() + ys.max()) / 2
    su, de = ys < mid, ys >= mid
    if su.sum() < 30 or de.sum() < 30:
        return None
    def fit(yy, xx):
        if xx.max() == xx.min():
            return float(np.median(yy)), 0.0
        a, b = np.polyfit(xx, yy, 1)
        return a * 256 + b, a
    ysup, _ = fit(ys[su], xs[su])
    ydeep, adeep = fit(ys[de], xs[de])
    return abs(ydeep - ysup) * (oh / 512.0), math.degrees(math.atan(adeep))


def fasc_angle(fb, min_sz=20):
    lbl, n = ndi.label(fb > 0)
    angs, wts = [], []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(xs) < min_sz:
            continue
        X = np.column_stack([xs - xs.mean(), ys - ys.mean()]).astype(float)
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        vx, vy = vt[0]
        a = math.degrees(math.atan2(vy, vx))
        a = a - 180 if a > 90 else (a + 180 if a < -90 else a)
        angs.append(a); wts.append(len(xs))
    if not angs:
        return None
    angs = np.array(angs); wts = np.array(wts, float)
    o = np.argsort(angs); angs, wts = angs[o], wts[o]
    cw = np.cumsum(wts)
    return float(angs[np.searchsorted(cw, cw[-1] / 2)])


test = sorted((ROOT / 'data/umud-challenge/test_images_v2').rglob('*.tif'))
rows = []
for k, p in enumerate(test):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    oh, ow = img.shape
    fb = (predict(fasc, img) > 0.5).astype(np.uint8)
    ab = (predict(apo, img) > 0.5).astype(np.uint8)
    ds = deep_sup(ab, oh)
    fa = fasc_angle(fb)
    rec = {'image_id': p.name, 'fasc_cov': float(fb.mean()), 'apo_cov': float(ab.mean())}
    if ds is not None and fa is not None:
        mt_px, dang = ds
        pa = abs(fa - dang)
        pa = 180 - pa if pa > 90 else pa
        rec['pa_raw'] = pa
        rec['mt_raw_mm'] = mt_px * PIXEL_TO_MM
        rec['fl_raw_mm'] = (mt_px / (math.sin(math.radians(max(pa, 0.5))) + 1e-6)) * PIXEL_TO_MM
    rows.append(rec)
    if (k + 1) % 50 == 0:
        print(f"  {k+1}/{len(test)}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(ROOT / 'data/dltrack-models/dltrack_raw_geom.csv', index=False)
n_valid = df[['pa_raw', 'mt_raw_mm', 'fl_raw_mm']].notna().all(1).sum()
print(f"\nRAW DL_Track geometry: {n_valid}/{len(df)} valid")
print(df[['pa_raw', 'mt_raw_mm', 'fl_raw_mm']].describe(percentiles=[.5]).round(2).loc[['mean', 'std', '50%', 'min', 'max']])
