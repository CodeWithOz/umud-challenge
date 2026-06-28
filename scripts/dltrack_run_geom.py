"""Run DL_Track's OWN doCalculations geometry on the test images using the
downloaded pretrained VGG16 models. No Kaggle GPU (CPU TensorFlow / tf_keras)."""
import os, sys, types
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
# Stub tkinter (only used for an error popup we won't hit)
tk = types.ModuleType('tkinter')
tk.messagebox = types.SimpleNamespace(showerror=lambda *a, **k: None)
sys.modules['tkinter'] = tk

import math
from pathlib import Path
import numpy as np, cv2, pandas as pd
import tf_keras as keras
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dltrack_do_calculations as dl

ROOT = Path('/Users/ucheozoemena/01-projects/umud-challenge')
fasc = keras.models.load_model(ROOT / 'data/dltrack-models/model-fasc-VGG16.h5', compile=False)
apo = keras.models.load_model(ROOT / 'data/dltrack-models/model-apo-VGG16.h5', compile=False)

# DL_Track default analysis parameters (from docstring example, current key names).
DIC = {
    "aponeurosis_detection_threshold": "0.15",
    "aponeurosis_length_threshold": "300",
    "fascicle_detection_threshold": "0.04",
    "fascicle_length_threshold": "20",
    "minimal_muscle_width": "30",
    "minimal_pennation_angle": "5",
    "maximal_pennation_angle": "45",
}
H = W = 512


def model_input(img):
    r = cv2.resize(img, (W, H))
    return (np.stack([r, r, r], -1).astype(np.float32) / 255.0)[None]


def run_one(img):
    x = model_input(img)
    out = dl.doCalculations(x, x[0].copy(), H, W, None, 10, apo, fasc, DIC, False, None)
    fasc_l, pennation, x_low, x_high, midthick, fig = out
    import matplotlib.pyplot as plt
    if fig is not None:
        plt.close(fig)
    if fasc_l is None or len(fasc_l) == 0:
        return None
    return (float(np.median(fasc_l)), float(np.median(pennation)),
            float(midthick) if midthick is not None else np.nan, len(fasc_l))


if __name__ == "__main__":
    test = sorted((ROOT / 'data/umud-challenge/test_images_v2').rglob('*.tif'))
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else len(test)
    rows = []
    for k, p in enumerate(test[:lim]):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        try:
            r = run_one(img)
        except Exception as e:
            r = None
            if k < 3:
                print(f"  {p.name} ERROR: {repr(e)[:160]}", flush=True)
        rec = {'image_id': p.name}
        if r is not None:
            rec.update(fl_px=r[0], pa_deg=r[1], mt_px=r[2], n_fasc=r[3])
        rows.append(rec)
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{lim}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / 'data/dltrack-models/dltrack_native_geom.csv', index=False)
    valid = df['pa_deg'].notna().sum() if 'pa_deg' in df else 0
    print(f"\nDL_Track NATIVE geometry: {valid}/{len(df)} valid")
    if valid:
        PX = 0.0881
        df['fl_mm'] = df['fl_px'] * PX
        df['mt_mm'] = df['mt_px'] * PX
        print(df[['pa_deg', 'fl_mm', 'mt_mm', 'n_fasc']].describe(percentiles=[.5]).round(2).loc[['mean', 'std', '50%', 'min', 'max']])
