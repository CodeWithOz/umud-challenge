"""Generate Block25 hidden-safe Block19 + SMP-PCA-geometry ensemble notebook.

Key change vs Block23: the SMP fascicle pennation angle is measured with
per-connected-component PCA (size-weighted median orientation), which recovers the
real ~13-16 deg fascicle angle instead of the flattened ~5 deg that whole-cloud
fitLine produces on stacked parallel fascicles. PA is then fascicle-angle minus
deep-aponeurosis-angle. The SMP per-image signal is z-scored and remapped to the
LB-confirmed centers / Block19-matched spreads, then ensembled into Block19.

Validated locally (251 test images): SMP-PCA PA correlates +0.28 with the
independent quickdirty estimator (naive PA: +0.03 -> noise); MT +0.29, FL +0.37.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_NB = ROOT / "notebooks/submission-blend-qdc-cxs5/submission-blend-qdc-cxs5.ipynb"
OUT_DIR = ROOT / "notebooks/submission-block31-dilate"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.split("\n")]}


def code(source: str) -> dict:
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src}


INSTALL_CELL = """import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "ttach", "segmentation-models-pytorch"],
    check=True,
)
"""


SMP_CELL = r'''import math
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import segmentation_models_pytorch as smp
import ttach as tta

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
cv2.setNumThreads(0)
warnings.filterwarnings("ignore")


class SMP_CFG:
    H = 640
    W = 960
    BATCH = 2
    BACKBONE = "efficientnet-b7"
    PIXEL_TO_MM = 0.0881
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


SMP_APO_PATH = Path("/kaggle/input/notebooks/ucheozoemena/umud-train-corrected-smp/best_aponeurosis_model.pth")
SMP_FASC_PATH = Path("/kaggle/input/notebooks/ucheozoemena/umud-train-fasc-dilate/best_fascicle_model.pth")
assert SMP_APO_PATH.exists(), SMP_APO_PATH
assert SMP_FASC_PATH.exists(), SMP_FASC_PATH


class TestImageDatasetSMP(Dataset):
    def __init__(self, test_dir: Path):
        self.images = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        path = self.images[idx]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        orig_h, orig_w = image.shape
        resized = cv2.resize(image, (SMP_CFG.W, SMP_CFG.H))
        tensor = torch.tensor(((resized / 255.0).astype(np.float32) - 0.5) / 0.5).unsqueeze(0)
        return tensor, path.name, orig_h, orig_w


def make_smp_model() -> torch.nn.Module:
    return smp.UnetPlusPlus(encoder_name=SMP_CFG.BACKBONE, encoder_weights=None, in_channels=1, classes=1)


def load_smp(path: Path) -> torch.nn.Module:
    model = make_smp_model()
    state = torch.load(path, map_location=SMP_CFG.DEVICE)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.to(SMP_CFG.DEVICE)
    model.eval()
    return model


def _deep_sup(apo_mask: np.ndarray):
    """Return (mt_px, deep_angle_deg) from the predicted aponeurosis region, or None."""
    ys, xs = np.where(apo_mask > 0)
    if len(ys) < 50:
        return None
    mid = (ys.min() + ys.max()) / 2.0
    deep_sel = ys >= mid
    sup_sel = ys < mid
    if deep_sel.sum() < 30 or sup_sel.sum() < 30:
        return None
    dy, dx = ys[deep_sel], xs[deep_sel]
    deep_ang = math.degrees(math.atan(np.polyfit(dx, dy, 1)[0])) if dx.max() > dx.min() else 0.0
    mt_px = abs(float(np.median(dy) - np.median(ys[sup_sel])))
    return mt_px, deep_ang


def _fasc_angle_pca(fasc_mask: np.ndarray, min_sz: int = 30):
    """Size-weighted median per-connected-component principal-direction angle (deg)."""
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
        angs.append(a)
        wts.append(len(xs))
    if not angs:
        return None
    angs = np.array(angs); wts = np.array(wts, float)
    order = np.argsort(angs)
    angs, wts = angs[order], wts[order]
    cw = np.cumsum(wts)
    return float(angs[np.searchsorted(cw, cw[-1] / 2.0)])


def extract_pca_px(apo_mask: np.ndarray, fasc_mask: np.ndarray) -> dict:
    pa_deg = np.nan
    mt_px = np.nan
    fl_px = np.nan
    status = "fallback"
    ds = _deep_sup(apo_mask)
    fang = _fasc_angle_pca(fasc_mask)
    if ds is not None:
        mt_px, deep_ang = ds
        status = "mt_ok"
        if fang is not None:
            pa = abs(fang - deep_ang)
            if pa > 90:
                pa = 180 - pa
            pa_deg = float(np.clip(pa, 1.0, 45.0))
            fl_px = mt_px / (math.sin(math.radians(max(pa_deg, 0.5))) + 1e-6)
            status = "mt_ok+pa_ok"
    return {"pa_deg": pa_deg, "fl_px": float(fl_px) if np.isfinite(fl_px) else np.nan,
            "mt_px": float(mt_px) if np.isfinite(mt_px) else np.nan, "status": status}


def run_smp_inference(test_dir: Path):
    apo_model = load_smp(SMP_APO_PATH)
    fasc_model = load_smp(SMP_FASC_PATH)
    transforms = tta.Compose([tta.HorizontalFlip(), tta.Multiply(factors=[0.9, 1.0, 1.1])])
    tta_apo = tta.SegmentationTTAWrapper(apo_model, transforms, merge_mode="mean")
    tta_fasc = tta.SegmentationTTAWrapper(fasc_model, transforms, merge_mode="mean")
    loader = DataLoader(TestImageDatasetSMP(test_dir), batch_size=SMP_CFG.BATCH, shuffle=False, num_workers=2)
    rows = []
    with torch.no_grad():
        for tensors, filenames, orig_hs, orig_ws in tqdm(loader, desc="smp inference"):
            tensors = tensors.to(SMP_CFG.DEVICE)
            apo_probs = torch.sigmoid(tta_apo(tensors)).squeeze(1).detach().cpu().numpy()
            fasc_probs = torch.sigmoid(tta_fasc(tensors)).squeeze(1).detach().cpu().numpy()
            for i, image_id in enumerate(filenames):
                oh, ow = int(orig_hs[i]), int(orig_ws[i])
                apo_mask = cv2.resize((apo_probs[i] > 0.5).astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST)
                fasc_mask = cv2.resize((fasc_probs[i] > 0.5).astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST)
                geom = extract_pca_px(apo_mask, fasc_mask)
                geom.update({"image_id": image_id, "apo_cov": float(apo_mask.mean()),
                             "fasc_cov": float(fasc_mask.mean()), "orig_h": oh, "orig_w": ow})
                rows.append(geom)
    debug = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    raw = pd.DataFrame({
        "image_id": debug["image_id"],
        "pa_deg": debug["pa_deg"],
        "fl_mm": debug["fl_px"] * SMP_CFG.PIXEL_TO_MM,
        "mt_mm": debug["mt_px"] * SMP_CFG.PIXEL_TO_MM,
    })
    return raw, debug


def zmap(series: pd.Series, center: float, out_std: float) -> pd.Series:
    """Z-score by the batch's own robust center/scale, remap to (center, out_std).
    NaN-safe: missing entries stay NaN (caller fills from Block19)."""
    x = series.astype(float)
    med = np.nanmedian(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-9:
        sd = 1.0
    z = (x - med) / sd
    return center + z * out_std


# ---- Block19 base (already computed as `submit`) ----
block19_df = submit[["image_id", "pa_deg", "fl_mm", "mt_mm"]].copy().sort_values("image_id").reset_index(drop=True)

# Anchors = Block19 medians (PA center confirmed optimal by Block24 probe), Block19 spreads.
CENTER = {"pa_deg": float(block19_df.pa_deg.median()),
          "fl_mm": float(block19_df.fl_mm.median()),
          "mt_mm": float(block19_df.mt_mm.median())}
SPREAD = {"pa_deg": float(block19_df.pa_deg.std()),
          "fl_mm": float(block19_df.fl_mm.std()),
          "mt_mm": float(block19_df.mt_mm.std())}

smp_raw, smp_debug = run_smp_inference(COMPETITION_DIR / "test_images_v2/test_set_v2")

# Map SMP per-image signal onto the proven Block19 marginals.
smp_mapped = pd.DataFrame({"image_id": smp_raw["image_id"]})
for col in ["pa_deg", "fl_mm", "mt_mm"]:
    smp_mapped[col] = zmap(smp_raw[col], CENTER[col], SPREAD[col]).values
smp_mapped = smp_mapped.sort_values("image_id").reset_index(drop=True)

# Ensemble: w * SMP-mapped + (1-w) * Block19, falling back to Block19 where SMP is NaN.
W = {"pa_deg": 0.30, "fl_mm": 0.45, "mt_mm": 0.55}  # block28: MT/FL signal strong (0.40), PA weak (0.21)
ens = block19_df.merge(smp_mapped, on="image_id", suffixes=("_b19", "_smp"), validate="one_to_one")
for col in ["pa_deg", "fl_mm", "mt_mm"]:
    s = ens[f"{col}_smp"]
    b = ens[f"{col}_b19"]
    blended = W[col] * s + (1.0 - W[col]) * b
    ens[col] = blended.where(s.notna(), b)

final = ens[["image_id", "pa_deg", "fl_mm", "mt_mm"]].sort_values("image_id").reset_index(drop=True)

# Hidden-safe safety net: any non-finite -> Block19 value -> global center.
final = final.merge(block19_df, on="image_id", suffixes=("", "_b19"))
for col in ["pa_deg", "fl_mm", "mt_mm"]:
    final[col] = final[col].where(np.isfinite(final[col]), final[f"{col}_b19"])
    final[col] = final[col].fillna(CENTER[col])
final = final[["image_id", "pa_deg", "fl_mm", "mt_mm"]]
final[["pa_deg", "fl_mm", "mt_mm"]] = final[["pa_deg", "fl_mm", "mt_mm"]].clip(
    lower=[1.0, 30.0, 8.0], upper=[45.0, 200.0, 50.0], axis=1)

assert len(final) == len(block19_df), (len(final), len(block19_df))
assert final[["pa_deg", "fl_mm", "mt_mm"]].notna().all().all()

final.to_csv("/kaggle/working/submission.csv", index=False)
block19_df.to_csv("/kaggle/working/submission_block19_component.csv", index=False)
smp_raw.to_csv("/kaggle/working/submission_smp_raw.csv", index=False)
smp_mapped.to_csv("/kaggle/working/submission_smp_mapped.csv", index=False)
smp_debug.to_csv("/kaggle/working/submission_smp_debug_px.csv", index=False)
ens.to_csv("/kaggle/working/submission_debug_ensemble.csv", index=False)

print("CENTER:", {k: round(v, 3) for k, v in CENTER.items()})
print("SPREAD:", {k: round(v, 3) for k, v in SPREAD.items()})
print("W:", W)
print("Rows:", len(final), "NaNs:", int(final.isna().sum().sum()))
print("SMP status counts:")
print(smp_debug["status"].value_counts())
print("SMP raw describe:")
print(smp_raw[["pa_deg", "fl_mm", "mt_mm"]].describe().round(3))
print("FINAL describe:")
print(final[["pa_deg", "fl_mm", "mt_mm"]].describe().round(3))
print(final.head())
'''


def main() -> None:
    nb = json.loads(BASE_NB.read_text())
    cells = nb["cells"]
    cells[0] = md(
        """# UMUD - Block25 Block19 + SMP-PCA-geometry ensemble

Hidden-safe. Runs the Block19 qdc+cxs5-refresh pipeline, mounts the Block20 SMP B7
weights, predicts apo+fascicle masks on the mounted test images, and measures
geometry with **per-connected-component PCA** for the fascicle angle (recovers the
real ~13-16 deg pennation instead of the flattened ~5 deg from whole-cloud line
fitting). The SMP per-image signal is z-scored to the proven Block19 marginals and
ensembled into Block19 (PA 0.50, FL 0.35, MT 0.40), falling back to Block19 / global
centers on any non-finite value.
"""
    )
    cells.extend(
        [
            md("## Install SMP inference dependencies"),
            code(INSTALL_CELL),
            md("## SMP-PCA geometry inference and ensemble"),
            code(SMP_CELL),
        ]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    }
    (OUT_DIR / "submission-block31-dilate.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": "ucheozoemena/umud-submission-block31-dilate",
        "title": "UMUD Submission Block31 Dilate",
        "code_file": "submission-block31-dilate.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [],
        "kernel_sources": [
            "ucheozoemena/umud-train-mounted-phase-3",
            "ucheozoemena/umud-train-apo-gray55-cxs5-refresh",
            "ucheozoemena/umud-train-corrected-smp",
            "ucheozoemena/umud-train-fasc-dilate",
        ],
        "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9",
        "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
