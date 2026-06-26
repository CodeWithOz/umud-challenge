"""Local SMP inference + geometry comparison (naive vs per-component PCA fascicle angle).

Runs the block20 trained apo+fasc B7 models on local test images, computes geometry
two ways, and reports per-image distributions. No Kaggle slot used.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd
import scipy.ndimage as ndi
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
WDIR = ROOT / "data/kaggle-outputs/block20-lakhindar-smp"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
H, W = 512, 768
PIXEL_TO_MM = 0.0881


def make():
    return smp.UnetPlusPlus(encoder_name="efficientnet-b7", encoder_weights=None, in_channels=1, classes=1)


def load(f):
    m = make(); m.load_state_dict(torch.load(WDIR / f, map_location="cpu"))
    return m.to(DEV).eval()


@torch.no_grad()
def pred(model, img):
    r = cv2.resize(img, (W, H))
    t = torch.tensor(((r / 255.0).astype(np.float32) - 0.5) / 0.5)[None, None].to(DEV)
    return torch.sigmoid(model(t))[0, 0].cpu().numpy()


def deep_sup_lines(apo_mask):
    """Return (sup_med_row, deep_med_row, deep_angle_deg) from predicted apo region."""
    ys, xs = np.where(apo_mask > 0)
    if len(ys) < 50:
        return None
    mid = (ys.min() + ys.max()) / 2
    out = {}
    for lab, sel in [("sup", ys < mid), ("deep", ys >= mid)]:
        yy, xx = ys[sel], xs[sel]
        if len(xx) < 30:
            return None
        ang = math.degrees(math.atan(np.polyfit(xx, yy, 1)[0])) if xx.max() > xx.min() else 0.0
        out[lab] = (float(np.median(yy)), ang)
    return out


def fasc_angle_naive(fasc_mask):
    pts = np.column_stack(np.where(fasc_mask > 0))
    if len(pts) < 50:
        return None
    [vx, vy, _, _] = cv2.fitLine(np.float32(np.flip(pts, axis=1)), cv2.DIST_L2, 0, 0.01, 0.01)
    return math.degrees(math.atan2(float(np.ravel(vy)[0]), float(np.ravel(vx)[0])))


def fasc_angle_pca(fasc_mask, min_sz=30):
    """Size-weighted median per-component principal-direction angle (deg from horizontal)."""
    lbl, n = ndi.label(fasc_mask > 0)
    angs, wts = [], []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(xs) < min_sz:
            continue
        X = np.column_stack([xs - xs.mean(), ys - ys.mean()]).astype(float)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        vx, vy = vt[0]
        a = math.degrees(math.atan2(vy, vx))
        if a > 90:
            a -= 180
        elif a < -90:
            a += 180
        angs.append(a); wts.append(len(xs))
    if not angs:
        return None
    angs = np.array(angs); wts = np.array(wts, float)
    order = np.argsort(angs)
    angs, wts = angs[order], wts[order]
    cw = np.cumsum(wts)
    return float(angs[np.searchsorted(cw, cw[-1] / 2)])


def main():
    apo_model = load("best_aponeurosis_model.pth")
    fasc_model = load("best_fascicle_model.pth")
    test = sorted(Path(ROOT / "data/umud-challenge/test_images_v2").rglob("*.tif"))
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else len(test)
    test = test[:lim]
    rows = []
    for k, p in enumerate(test):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        oh, ow = img.shape
        ap = (pred(apo_model, img) > 0.5).astype(np.uint8)
        fa = (pred(fasc_model, img) > 0.5).astype(np.uint8)
        ap = cv2.resize(ap, (ow, oh), interpolation=cv2.INTER_NEAREST)
        fa = cv2.resize(fa, (ow, oh), interpolation=cv2.INTER_NEAREST)
        ds = deep_sup_lines(ap)
        a_naive = fasc_angle_naive(fa)
        a_pca = fasc_angle_pca(fa)
        rec = {"image_id": p.name, "oh": oh, "ow": ow,
               "fasc_cov": float(fa.mean()), "apo_cov": float(ap.mean())}
        if ds is not None:
            sup_r, sup_a = ds["sup"]; deep_r, deep_a = ds["deep"]
            mt_px = abs(deep_r - sup_r)
            rec["mt_mm"] = mt_px * PIXEL_TO_MM
            rec["deep_ang"] = deep_a
            for tag, fang in [("naive", a_naive), ("pca", a_pca)]:
                if fang is not None:
                    pa = abs(fang - deep_a)
                    if pa > 90:
                        pa = 180 - pa
                    rec[f"pa_{tag}"] = pa
                    rec[f"fl_{tag}"] = (mt_px / (math.sin(math.radians(max(pa, 0.5))) + 1e-6)) * PIXEL_TO_MM
        rows.append(rec)
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(test)}", flush=True)
    df = pd.DataFrame(rows)
    outp = ROOT / "data/kaggle-outputs/block24-pa-center-probe/local_smp_geom.csv"
    df.to_csv(outp, index=False)
    print("\nwrote", outp, "rows", len(df))
    for c in ["pa_naive", "pa_pca", "fl_naive", "fl_pca", "mt_mm", "deep_ang"]:
        if c in df:
            s = df[c].dropna()
            print(f"{c:9s} n={len(s):3d} med={s.median():7.2f} mean={s.mean():7.2f} "
                  f"p25={s.quantile(.25):7.2f} p75={s.quantile(.75):7.2f} std={s.std():6.2f}")


if __name__ == "__main__":
    main()
