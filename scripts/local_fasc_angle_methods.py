"""Compare fascicle-angle estimators on block20 SMP predictions (local, no slot).
Metric: Spearman agreement with the independent quickdirty PA on the same images.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np, cv2, pandas as pd
import scipy.ndimage as ndi
from scipy.stats import spearmanr
import torch, segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
WDIR = ROOT / "data/kaggle-outputs/block20-lakhindar-smp"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
H, W = 512, 768


def load(f):
    m = smp.UnetPlusPlus(encoder_name="efficientnet-b7", encoder_weights=None, in_channels=1, classes=1)
    m.load_state_dict(torch.load(WDIR / f, map_location="cpu"))
    return m.to(DEV).eval()


@torch.no_grad()
def prob(model, img):
    r = cv2.resize(img, (W, H))
    t = torch.tensor(((r / 255.0).astype(np.float32) - 0.5) / 0.5)[None, None].to(DEV)
    return torch.sigmoid(model(t))[0, 0].cpu().numpy()


def deep_angle(apo_bin):
    ys, xs = np.where(apo_bin > 0)
    if len(ys) < 50:
        return 0.0
    mid = (ys.min() + ys.max()) / 2
    sel = ys >= mid
    if sel.sum() < 30 or xs[sel].max() == xs[sel].min():
        return 0.0
    return math.degrees(math.atan(np.polyfit(xs[sel], ys[sel], 1)[0]))


def m_comp_pca(fb, min_sz=30, elong=1.0):
    lbl, n = ndi.label(fb > 0)
    angs, wts = [], []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(xs) < min_sz:
            continue
        X = np.column_stack([xs - xs.mean(), ys - ys.mean()]).astype(float)
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        if s[1] < 1e-6 or (s[0] / (s[1] + 1e-9)) < elong:
            continue
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


def m_structure_tensor(fp):
    """Probability-weighted structure tensor dominant orientation (deg from horiz)."""
    P = fp.astype(np.float64)
    gy, gx = np.gradient(P)
    w = P
    Jxx = float((w * gx * gx).sum())
    Jyy = float((w * gy * gy).sum())
    Jxy = float((w * gx * gy).sum())
    # orientation of structures = eigenvector of smaller eigenvalue
    theta = 0.5 * math.atan2(2 * Jxy, Jxx - Jyy)
    # theta is the dominant gradient orientation; structure runs perpendicular
    ang = math.degrees(theta) + 90.0
    ang = ang - 180 if ang > 90 else (ang + 180 if ang < -90 else ang)
    return ang


def main():
    apo_model = load("best_aponeurosis_model.pth")
    fasc_model = load("best_fascicle_model.pth")
    test = sorted(Path(ROOT / "data/umud-challenge/test_images_v2").rglob("*.tif"))
    rows = []
    for k, p in enumerate(test):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        ap = prob(apo_model, img) > 0.5
        fp = prob(fasc_model, img)
        fb = fp > 0.5
        dang = deep_angle(ap.astype(np.uint8))
        def pa(fang):
            if fang is None:
                return np.nan
            v = abs(fang - dang)
            return 180 - v if v > 90 else v
        rows.append({
            "image_id": p.name,
            "pa_pca": pa(m_comp_pca(fb)),
            "pa_pca_elong3": pa(m_comp_pca(fb, elong=3.0)),
            "pa_st": pa(m_structure_tensor(fp)),
        })
        if (k + 1) % 60 == 0:
            print(f"  {k+1}/{len(test)}", flush=True)
    df = pd.DataFrame(rows)
    qd = pd.read_csv(ROOT / "data/kaggle-outputs/block19-cxs5-refresh/submit/submission_quickdirty_raw_debug.csv")[["image_id", "pa_deg"]].rename(columns={"pa_deg": "pa_qd"})
    m = df.merge(qd, on="image_id")
    print("\nAgreement with independent quickdirty PA (Spearman):")
    for c in ["pa_pca", "pa_pca_elong3", "pa_st"]:
        d = m[[c, "pa_qd"]].dropna()
        print(f"  {c:14s} n={len(d):3d}  Spearman={spearmanr(d[c], d['pa_qd']).correlation:+.3f}  med={d[c].median():.2f}")
    df.to_csv(ROOT / "data/kaggle-outputs/block24-pa-center-probe/local_fasc_angle_methods.csv", index=False)


if __name__ == "__main__":
    main()
