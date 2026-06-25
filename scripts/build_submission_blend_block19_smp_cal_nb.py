"""Generate Block23 hidden-safe Block19 + calibrated SMP blend notebook."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_NB = ROOT / "notebooks/submission-blend-qdc-cxs5/submission-blend-qdc-cxs5.ipynb"
OUT_DIR = ROOT / "notebooks/submission-blend-block19-smpcal"


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
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "ttach",
        "segmentation-models-pytorch",
    ],
    check=True,
)
"""


SMP_CELL = """import math
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import segmentation_models_pytorch as smp
import ttach as tta

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
cv2.setNumThreads(0)
warnings.filterwarnings("ignore")


class SMP_CFG:
    H = 512
    W = 768
    BATCH = 2
    BACKBONE = "efficientnet-b7"
    PIXEL_TO_MM = 0.0881
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


SMP_KERNEL_DIR = Path("/kaggle/input/notebooks/ucheozoemena/umud-submission-lakhindar-smp")
SMP_APO_PATH = SMP_KERNEL_DIR / "best_aponeurosis_model.pth"
SMP_FASC_PATH = SMP_KERNEL_DIR / "best_fascicle_model.pth"
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
    return smp.UnetPlusPlus(
        encoder_name=SMP_CFG.BACKBONE,
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )


def load_smp(path: Path) -> torch.nn.Module:
    model = make_smp_model()
    state = torch.load(path, map_location=SMP_CFG.DEVICE)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.to(SMP_CFG.DEVICE)
    model.eval()
    return model


def extract_measurements_px(apo_mask: np.ndarray, fasc_mask: np.ndarray, orig_h: int, orig_w: int) -> dict:
    mid_y = orig_h // 2
    sup_mask = apo_mask.copy()
    deep_mask = apo_mask.copy()
    sup_mask[mid_y:, :] = 0
    deep_mask[:mid_y, :] = 0

    sup_pts = np.column_stack(np.where(sup_mask > 0))
    deep_pts = np.column_stack(np.where(deep_mask > 0))
    fasc_pts = np.column_stack(np.where(fasc_mask > 0))

    pa_deg = 15.0
    mt_px = 150.0
    status = "fallback"

    if len(sup_pts) > 50 and len(deep_pts) > 50:
        mt_px = abs(float(np.median(deep_pts[:, 0]) - np.median(sup_pts[:, 0])))
        status = "mt_ok"

    if len(fasc_pts) > 50 and len(deep_pts) > 50:
        try:
            [vx_f, vy_f, _, _] = cv2.fitLine(
                np.float32(np.flip(fasc_pts, axis=1)), cv2.DIST_L2, 0, 0.01, 0.01
            )
            [vx_d, vy_d, _, _] = cv2.fitLine(
                np.float32(np.flip(deep_pts, axis=1)), cv2.DIST_L2, 0, 0.01, 0.01
            )
            if vx_f[0] < 0:
                vx_f, vy_f = -vx_f, -vy_f
            if vx_d[0] < 0:
                vx_d, vy_d = -vx_d, -vy_d
            dot_product = np.clip((vx_f[0] * vx_d[0]) + (vy_f[0] * vy_d[0]), -1.0, 1.0)
            pa_deg = math.degrees(math.acos(float(dot_product)))
            if pa_deg > 90:
                pa_deg = 180 - pa_deg
            pa_deg = float(np.clip(pa_deg, 5.0, 45.0))
            status = status + "+pa_ok"
        except Exception:
            status = status + "+pa_fallback"

    fl_px = mt_px / (math.sin(math.radians(pa_deg)) + 1e-6)
    return {
        "pa_deg": float(pa_deg),
        "fl_px": float(fl_px),
        "mt_px": float(mt_px),
        "status": status,
    }


def run_smp_inference(test_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                orig_h = int(orig_hs[i])
                orig_w = int(orig_ws[i])
                apo_mask = cv2.resize((apo_probs[i] > 0.5).astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                fasc_mask = cv2.resize((fasc_probs[i] > 0.5).astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                geom = extract_measurements_px(apo_mask, fasc_mask, orig_h, orig_w)
                geom.update(
                    {
                        "image_id": image_id,
                        "apo_cov": float(apo_mask.mean()),
                        "fasc_cov": float(fasc_mask.mean()),
                        "orig_h": orig_h,
                        "orig_w": orig_w,
                    }
                )
                rows.append(geom)
    debug = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    raw = pd.DataFrame(
        {
            "image_id": debug["image_id"],
            "pa_deg": debug["pa_deg"],
            "fl_mm": debug["fl_px"] * SMP_CFG.PIXEL_TO_MM,
            "mt_mm": debug["mt_px"] * SMP_CFG.PIXEL_TO_MM,
        }
    )
    raw[["pa_deg", "fl_mm", "mt_mm"]] = raw[["pa_deg", "fl_mm", "mt_mm"]].clip(
        lower=[5.0, 30.0, 10.0],
        upper=[45.0, 200.0, 50.0],
        axis=1,
    )
    return raw, debug


def calibrate_smp(raw: pd.DataFrame) -> pd.DataFrame:
    # Fixed from Block20 v1 public-test output and Block19 hidden-safe center.
    raw_med = {"pa_deg": 7.053608330646576, "fl_mm": 197.2364024869034, "mt_mm": 28.48902285714284}
    center = {"pa_deg": 16.783151381294555, "fl_mm": 75.32733023021203, "mt_mm": 20.38789389731057}
    alpha = {"pa_deg": 0.50, "fl_mm": 0.15, "mt_mm": 0.20}
    out = raw.copy()
    for col in ["pa_deg", "fl_mm", "mt_mm"]:
        out[col] = center[col] + alpha[col] * (raw[col] - raw_med[col])
    out[["pa_deg", "fl_mm", "mt_mm"]] = out[["pa_deg", "fl_mm", "mt_mm"]].clip(
        lower=[5.0, 30.0, 10.0],
        upper=[45.0, 200.0, 50.0],
        axis=1,
    )
    return out


block19_df = submit[["image_id", "pa_deg", "fl_mm", "mt_mm"]].copy().sort_values("image_id").reset_index(drop=True)
smp_raw, smp_debug = run_smp_inference(COMPETITION_DIR / "test_images_v2/test_set_v2")
smp_cal = calibrate_smp(smp_raw).sort_values("image_id").reset_index(drop=True)

blend = block19_df.merge(smp_cal, on="image_id", suffixes=("_block19", "_smpcal"), validate="one_to_one")
SMPCAL_WEIGHT = 0.35
BLOCK19_WEIGHT = 1.0 - SMPCAL_WEIGHT
for col in ["pa_deg", "fl_mm", "mt_mm"]:
    blend[col] = BLOCK19_WEIGHT * blend[f"{col}_block19"] + SMPCAL_WEIGHT * blend[f"{col}_smpcal"]

final = blend[["image_id", "pa_deg", "fl_mm", "mt_mm"]].sort_values("image_id")
assert len(final) == len(block19_df) == len(smp_cal), (len(final), len(block19_df), len(smp_cal))
assert final[["pa_deg", "fl_mm", "mt_mm"]].notna().all().all()

final.to_csv("/kaggle/working/submission.csv", index=False)
block19_df.to_csv("/kaggle/working/submission_block19_component.csv", index=False)
smp_raw.to_csv("/kaggle/working/submission_smp_raw.csv", index=False)
smp_cal.to_csv("/kaggle/working/submission_smp_calibrated.csv", index=False)
smp_debug.to_csv("/kaggle/working/submission_smp_debug_px.csv", index=False)
blend.to_csv("/kaggle/working/submission_debug_blend.csv", index=False)

print(f"Overwrote submission.csv with Block19/SMP-cal blend: block19={BLOCK19_WEIGHT:.2f}, smpcal={SMPCAL_WEIGHT:.2f}")
print("Rows:", len(final), "NaNs:", int(final.isna().sum().sum()))
print("SMP status counts:")
print(smp_debug["status"].value_counts())
display(final[["pa_deg", "fl_mm", "mt_mm"]].describe().round(3))
display(smp_cal[["pa_deg", "fl_mm", "mt_mm"]].describe().round(3))
display(final.head())
"""


def main() -> None:
    nb = json.loads(BASE_NB.read_text())
    cells = nb["cells"]
    cells[0] = md(
        """# UMUD - Block23 Block19 + Calibrated SMP Blend

Runs the hidden-safe Block19 qdc+cxs5-refresh pipeline, mounts the trained
Block20 SMP B7 weights, performs inference on the mounted test images, calibrates
the SMP geometry toward the current best notebook center, and writes:

`submission.csv = 0.65 * block19 + 0.35 * calibrated_smp`.
"""
    )
    cells.extend(
        [
            md("## Install SMP inference dependencies"),
            code(INSTALL_CELL),
            md("## Calibrated SMP inference and final blend"),
            code(SMP_CELL),
        ]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    }
    (OUT_DIR / "submission-blend-block19-smpcal.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": "ucheozoemena/umud-submission-blend-block19-smpcal",
        "title": "UMUD Submission Blend Block19 SMPCal",
        "code_file": "submission-blend-block19-smpcal.ipynb",
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
            "ucheozoemena/umud-submission-lakhindar-smp",
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
