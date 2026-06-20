"""Generate notebooks/train-apo-gray55/train-apo-gray55-phase-3.ipynb — train on gray55 prep dataset."""
import json
from pathlib import Path

BUILD_TRAIN_RUN = 18

FASTAI_RESNETS = frozenset({"resnet18", "resnet34", "resnet50"})

DATASET_SLUG_BY_RUN = {
    1: "ucheozoemena/umud-aligned-apo-gray55-timing-50",
    2: "ucheozoemena/umud-aligned-apo-gray55-timing-200",
    3: "ucheozoemena/umud-aligned-apo-gray55-timing-524",
    4: "ucheozoemena/umud-aligned-apo-gray55-full",
    5: "ucheozoemena/umud-aligned-apo-gray55-line-timing-50",
    6: "ucheozoemena/umud-aligned-apo-gray55-line-full",
    7: "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
    8: "ucheozoemena/umud-aligned-apo-gray55-line-timing-524",
    9: "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
    10: "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
    11: "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
    12: "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
    13: "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
    14: "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
    15: "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
    16: "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
    17: "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
    18: "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
    19: "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
}

EXTRA_DATASET_SOURCES_BY_RUN: dict[int, list[str]] = {
    12: ["ucheozoemena/umud-apo-line-model-200"],
}

SCRIPTS_DIR = Path(__file__).resolve().parent


def embed_script(name: str) -> str:
    return SCRIPTS_DIR.joinpath(name).read_text()


def md(source: str) -> dict:
    lines = source.split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in lines]}


def code(source: str) -> dict:
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": src,
    }


cells: list[dict] = [
    md(
        """# UMUD — Train Apo U-Net on Gray55 Prep Dataset

**GPU notebook** — mounts a **gray55 apo prep** dataset. Images were preprocessed with RGB(55,55,55) outside the ultrasound bbox before 256px resize.

> Edit *Configuration*, then re-run from there downward."""
    ),
    md("## Configuration"),
    code(
        """# --- Parameters you can change ---
RANDOM_SEED = 42
TRAIN_RUN = 18  # Block 7 encoder sweep — see TRAIN_PROFILES

VALID_PCT = 0.20
STRATIFY_VAL_BY_RESOLUTION = True  # uses manifest resolution_cohort when True
BATCH_SIZE = 8
IMG_SIZE = 256
APO_FULL = 1044
FULL_EPOCHS = 10
MM_PER_PIXEL = 0.075  # production calibration for val UMUD score

USE_CLASS_WEIGHTS = True
APO_FG_WEIGHT = 15.0

TRAIN_PROFILES = {
    1: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-timing-50",
        "epochs": 1,
        "label": "GAT1 gray55 apo 50×1ep",
        "export_name": "apo_gray55_baseline.pkl",
    },
    2: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-timing-200",
        "epochs": 1,
        "label": "GAT2 gray55 apo 200×1ep",
        "export_name": "apo_gray55_baseline.pkl",
    },
    3: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-timing-524",
        "epochs": FULL_EPOCHS // 2,
        "label": "GAT3 gray55 apo 524×5ep",
        "export_name": "apo_gray55_baseline.pkl",
    },
    4: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-full",
        "epochs": FULL_EPOCHS,
        "label": "GAT4 gray55 apo full 1044×10ep",
        "export_name": "apo_gray55_baseline.pkl",
    },
    5: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-50",
        "epochs": 5,
        "label": "GAT5 gray55+line apo 50×5ep micro",
        "export_name": "apo_gray55_line_baseline.pkl",
    },
    6: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-full",
        "epochs": FULL_EPOCHS,
        "label": "GAT6 gray55+line apo full 1044×10ep",
        "export_name": "apo_gray55_line_baseline.pkl",
    },
    7: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
        "epochs": 5,
        "label": "GAT7 gray55+line apo 200×5ep stratified val",
        "export_name": "apo_gray55_line_200.pkl",
    },
    8: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-524",
        "epochs": 5,
        "label": "GAT8 gray55+line apo 524×5ep stratified val",
        "export_name": "apo_gray55_line_524.pkl",
    },
    9: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
        "epochs": FULL_EPOCHS,
        "label": "GAT9 gray55+line apo 200×10ep stratified val",
        "export_name": "apo_gray55_line_200_10ep.pkl",
    },
    10: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
        "epochs": 8,
        "arch": "resnet34",
        "label": "GAT10 gray55+line apo 200×8ep stratified val",
        "export_name": "apo_gray55_line_200_8ep.pkl",
    },
    11: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
        "epochs": 5,
        "arch": "resnet50",
        "label": "GAT11 gray55+line apo 200×5ep resnet50 (Block 6c)",
        "export_name": "apo_gray55_line_200_r50.pkl",
    },
    12: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
        "epochs": 0,
        "arch": "resnet34",
        "eval_only": True,
        "label": "GAT12 val UMUD backfill — prod r34 200×5ep (no retrain)",
        "export_name": "apo_gray55_line_200.pkl",
    },
    13: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
        "epochs": 5,
        "arch": "resnet18",
        "label": "GAT13 Block7 resnet18 200×5ep",
        "export_name": "apo_gray55_line_200_r18.pkl",
    },
    14: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
        "epochs": 5,
        "arch": "convnext_tiny",
        "label": "GAT14 Block7 convnext_tiny 200×5ep",
        "export_name": "apo_gray55_line_200_cxt.pkl",
    },
    15: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
        "epochs": 5,
        "arch": "convnext_small",
        "label": "GAT15 Block7 convnext_small 200×5ep",
        "export_name": "apo_gray55_line_200_cxs.pkl",
    },
    16: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
        "epochs": 5,
        "arch": "efficientnet_b0",
        "label": "GAT16 Block7 efficientnet_b0 200×5ep",
        "export_name": "apo_gray55_line_200_enb0.pkl",
    },
    17: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
        "epochs": 5,
        "arch": "efficientnet_b1",
        "label": "GAT17 Block7 efficientnet_b1 200×5ep",
        "export_name": "apo_gray55_line_200_enb1.pkl",
    },
    18: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
        "epochs": 5,
        "arch": "mobilenetv3_small_100",
        "label": "GAT18 Block7 mobilenetv3_small 200×5ep",
        "export_name": "apo_gray55_line_200_mnv3.pkl",
    },
    19: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
        "epochs": 5,
        "arch": "regnetx_004",
        "label": "GAT19 Block7 regnetx_004 200×5ep",
        "export_name": "apo_gray55_line_200_rgx004.pkl",
    },
}

profile = TRAIN_PROFILES[TRAIN_RUN]
DATASET_SLUG = profile["dataset_slug"]
EPOCHS = profile["epochs"]
ARCH = profile.get("arch", "resnet34")
EVAL_ONLY = profile.get("eval_only", False)
USE_TIMM = ARCH not in ("resnet18", "resnet34", "resnet50")
EXPORT_NAME = profile["export_name"]
print(f"TRAIN_RUN={TRAIN_RUN} | {profile['label']} | arch={ARCH} | timm={USE_TIMM} | eval_only={EVAL_ONLY} | dataset={DATASET_SLUG} | epochs={EPOCHS}")
"""
    ),
    code(
        """from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import kagglehub
from fastai.vision.all import (
    AddMaskCodes,
    CrossEntropyLossFlat,
    Dice,
    IntToFloatTensor,
    PILImage,
    PILMask,
    RandomSplitter,
    Resize,
    TransformBlock,
    aug_transforms,
    get_image_files,
    resnet18,
    resnet34,
    resnet50,
    unet_learner,
)
from fastai.data.block import DataBlock
from fastai.data.transforms import IndexSplitter

DATASET_ROOT = Path(f"/kaggle/input/datasets/{DATASET_SLUG}")
if not DATASET_ROOT.exists():
    DATASET_ROOT = Path(kagglehub.dataset_download(DATASET_SLUG))
WORKING = Path("/kaggle/working")

print(f"Dataset root: {DATASET_ROOT} (exists={DATASET_ROOT.exists()})")
"""
    ),
    code(
        """def resolve_subdir(root: Path, name: str) -> Path:
    direct = root / name
    if direct.exists():
        return direct
    candidates = [p for p in root.rglob(name) if p.is_dir() and p.name == name]
    if not candidates:
        raise FileNotFoundError(f"Could not find {name}/ under {root}")
    return candidates[0]

IMG_DIR = resolve_subdir(DATASET_ROOT, "images")
MSK_DIR = resolve_subdir(DATASET_ROOT, "masks")
print(f"images: {IMG_DIR}")
print(f"masks: {MSK_DIR}")
"""
    ),
    code(
        """SEG_CODES = ["background", "structure"]

FASTAI_ENCODERS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}


def open_image_pil(fn):
    gray = np.array(PILImage.create(fn))
    if gray.ndim == 3:
        gray = gray[..., 0]
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    return PILImage.create(rgb)


def open_mask_pil(fn):
    arr = np.array(PILImage.create(fn))
    if arr.ndim == 3:
        arr = arr[..., 0]
    binary = (arr > 0).astype(np.uint8)
    return PILMask.create(binary)


def stratified_train_valid_stems(
    stems: list[str],
    labels: list[str],
    valid_pct: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    from collections import Counter
    import random

    counts = Counter(labels)
    # sklearn needs ≥2 per class; bucket rare native resolutions together.
    collapsed = [label if counts[label] >= 2 else "other" for label in labels]
    counts = Counter(collapsed)
    if min(counts.values()) < 2:
        collapsed = [
            label if counts[label] >= 2 else "other_bucket" for label in collapsed
        ]
        counts = Counter(collapsed)
    if min(counts.values()) >= 2:
        from sklearn.model_selection import train_test_split

        return train_test_split(
            stems,
            test_size=valid_pct,
            random_state=seed,
            stratify=collapsed,
        )

    rng = random.Random(seed)
    by_cohort: dict[str, list[str]] = {}
    for stem, label in zip(stems, labels):
        by_cohort.setdefault(label, []).append(stem)
    train: list[str] = []
    valid: list[str] = []
    for cohort_stems in by_cohort.values():
        rng.shuffle(cohort_stems)
        if len(cohort_stems) == 1:
            train.extend(cohort_stems)
            continue
        n_val = max(1, round(len(cohort_stems) * valid_pct))
        valid.extend(cohort_stems[:n_val])
        train.extend(cohort_stems[n_val:])
    print("Stratified sklearn failed — used per-cohort manual split")
    return train, valid


def make_dls(fnames, valid_pct=0.20, bs=8, seed=42, stratify_cohort: dict[str, str] | None = None):
    if stratify_cohort:
        stems = [Path(f).stem for f in fnames]
        labels = [stratify_cohort.get(s, "unknown") for s in stems]
        train_stems, valid_stems = stratified_train_valid_stems(
            stems, labels, valid_pct=valid_pct, seed=seed
        )
        valid_set = set(valid_stems)
        valid_idx = [i for i, s in enumerate(stems) if s in valid_set]
        splitter = IndexSplitter(valid_idx)
        print(
            f"Stratified val: {len(valid_stems)} images across "
            f"{len(set(labels))} resolution cohorts"
        )
    else:
        splitter = RandomSplitter(valid_pct=valid_pct, seed=seed)

    block = DataBlock(
        blocks=(
            TransformBlock(type_tfms=open_image_pil, batch_tfms=IntToFloatTensor),
            TransformBlock(
                type_tfms=open_mask_pil,
                item_tfms=AddMaskCodes(codes=SEG_CODES),
                batch_tfms=IntToFloatTensor,
            ),
        ),
        get_items=lambda _: fnames,
        get_x=lambda f: IMG_DIR / f.name,
        get_y=lambda f: MSK_DIR / f.name,
        splitter=splitter,
        item_tfms=Resize(IMG_SIZE),
        batch_tfms=aug_transforms(size=IMG_SIZE, min_scale=0.75, flip_vert=False, do_flip=True),
    )
    return block.dataloaders(fnames, bs=bs, num_workers=2)


def load_cohort_by_stem(root: Path) -> dict[str, str]:
    manifest_dir = resolve_subdir(root, "manifests")
    manifest_path = manifest_dir / "train_apo_gray55_line.csv"
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path} — falling back to random val split")
        return {}
    manifest = pd.read_csv(manifest_path)
    if "resolution_cohort" in manifest.columns:
        col = "resolution_cohort"
    elif {"img_h", "img_w"}.issubset(manifest.columns):
        manifest["resolution_cohort"] = manifest.apply(
            lambda r: f"{int(r.img_h)}x{int(r.img_w)}", axis=1
        )
        col = "resolution_cohort"
    else:
        print("Manifest missing resolution columns — random val split")
        return {}
    return dict(zip(manifest["stem"].astype(str), manifest[col].astype(str)))


img_fnames = get_image_files(IMG_DIR)
msk_lookup = {p.name for p in get_image_files(MSK_DIR)}
fnames = [f for f in img_fnames if f.name in msk_lookup]
print(f"Pairs: {len(fnames)}")
assert len(fnames) > 0, "No image/mask pairs in mounted dataset"
cohort_by_stem = load_cohort_by_stem(DATASET_ROOT) if STRATIFY_VAL_BY_RESOLUTION else {}
"""
    ),
    code(
        """t0 = time.perf_counter()
dls = make_dls(
    fnames,
    valid_pct=VALID_PCT,
    bs=BATCH_SIZE,
    seed=RANDOM_SEED,
    stratify_cohort=cohort_by_stem or None,
)
_ = dls.one_batch()
print(f"Dataloader ready: {time.perf_counter() - t0:.1f}s")
dls.show_batch(max_n=4)
"""
    ),
    md("## Timm U-Net helper (ConvNeXt, EfficientNet, …)"),
    code(embed_script("timm_unet.py")),
    code(
        """t_train = time.perf_counter()
import torch
from fastai.vision.all import load_learner


def resolve_pkl(preferred: list[Path], filename: str) -> Path:
    for p in preferred:
        if p.exists():
            return p
    hits = sorted(Path("/kaggle/input").rglob(filename))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Could not find {filename} under /kaggle/input")


if EVAL_ONLY:
    import kagglehub

    preferred = [
        Path("/kaggle/input/datasets/ucheozoemena/umud-apo-line-model-200") / EXPORT_NAME,
        Path("/kaggle/input/notebooks/ucheozoemena/umud-train-apo-gray55-phase-3") / EXPORT_NAME,
    ]
    try:
        apo_path = resolve_pkl(preferred, EXPORT_NAME)
    except FileNotFoundError:
        model_root = Path(kagglehub.dataset_download("ucheozoemena/umud-apo-line-model-200"))
        hits = sorted(model_root.rglob(EXPORT_NAME))
        if not hits:
            raise FileNotFoundError(f"Could not find {EXPORT_NAME} via mount or kagglehub")
        apo_path = hits[0]
    learn = load_learner(apo_path)
    train_sec = 0.0
    val_dice = float("nan")
    print(f"Eval-only: loaded {apo_path}")
else:
    if USE_TIMM:
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    if USE_CLASS_WEIGHTS:
        loss_weights = torch.tensor([1.0, APO_FG_WEIGHT])
        loss_func = CrossEntropyLossFlat(axis=1, weight=loss_weights)
        print(f"Class weights: background=1.0, structure={APO_FG_WEIGHT}")
    else:
        loss_func = CrossEntropyLossFlat(axis=1)

    if USE_TIMM:
        learn = timm_unet_learner(
            dls,
            ARCH,
            metrics=[Dice()],
            loss_func=loss_func,
            bottleneck="conv",
        )
    else:
        learn = unet_learner(
            dls,
            FASTAI_ENCODERS[ARCH],
            metrics=[Dice()],
            loss_func=loss_func,
            self_attention=True,
        )
    learn.fine_tune(EPOCHS)
    t1 = time.perf_counter()
    train_sec = t1 - t_train
    print(f"Train wall-clock: {train_sec:.1f}s")
    learn.export(WORKING / EXPORT_NAME)

    val_losses, val_metrics = learn.validate(dl=dls.valid)
    if isinstance(val_metrics, (list, tuple)):
        val_dice = float(val_metrics[0]) if val_metrics else float("nan")
    else:
        val_dice = float(val_metrics)
    print(f"Val Dice (reference): {val_dice:.4f}")

timing = pd.DataFrame(
    [
        {
            "train_run": TRAIN_RUN,
            "arch": ARCH,
            "eval_only": EVAL_ONLY,
            "n_pairs": len(fnames),
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "val_dice": round(val_dice, 4) if val_dice == val_dice else None,
            "total_sec": round(train_sec, 1),
            "sec_per_pair_epoch": round(train_sec / max(1, len(fnames) * max(EPOCHS, 1)), 3),
            "dataset": DATASET_SLUG,
        }
    ]
)
timing.to_csv(WORKING / "timing_report.csv", index=False)
display(timing)
"""
    ),
    md(
        """## Val UMUD score (primary model-selection metric)

End-to-end **segment-then-measure** on the stratified val split:

- **GT:** stretch-aligned fasc + line-converted apo masks → PA/FL/MT @ `MM_PER_PIXEL`
- **Pred:** production **fasc** model + **trained apo** (gray55 infer, horiz_parallel)
- **Metric:** official UMUD score (`scripts/umud_score.py`) — **lower is better**

Also reports `val_mt_ok_pct` (% val images with finite PA/FL/MT) — must reach **100%** before test submit."""
    ),
    code(embed_script("segment_geometry.py") + "\n\n" + embed_script("umud_score.py")),
    code(
        """from fastai.vision.all import load_learner
from tqdm.auto import tqdm
import kagglehub

COMPETITION_DIR = Path(
    "/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data"
)
if not COMPETITION_DIR.exists():
    COMPETITION_DIR = Path(
        kagglehub.competition_download("umud-challenge-muscle-architecture-in-ultrasound-data")
    )
COMP_DIRS = {
    "apo_img": COMPETITION_DIR / "apo_imgs_v1/apo_images_new_model_v1",
    "apo_mask": COMPETITION_DIR / "apo_masks_v1/apo_masks_new_model_v1",
    "fasc_img": COMPETITION_DIR / "fasc_imgs_v1/fasc_images_new_model_v1",
    "fasc_mask": COMPETITION_DIR / "fasc_masks_v1/fasc_masks_new_model_v1",
}
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def build_lookup(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        return {}
    return {
        p.name: p
        for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.name != "Thumbs.db"
    }


def resolve_pkl(preferred: list[Path], filename: str) -> Path:
    for p in preferred:
        if p.exists():
            return p
    hits = sorted(Path("/kaggle/input").rglob(filename))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Could not find {filename} under /kaggle/input")


def resolve_filename(stem: str, stem_to_filename: dict[str, str], lookups: dict[str, dict[str, Path]]) -> str | None:
    filename = stem_to_filename.get(stem)
    if filename and filename in lookups["apo_img"]:
        return filename
    for cand in (f"{stem}.tif", f"{stem}.png", f"{stem}.jpg"):
        if cand in lookups["apo_img"]:
            return cand
    return None


comp_lookups = {k: build_lookup(v) for k, v in COMP_DIRS.items()}
print({k: len(v) for k, v in comp_lookups.items()})
assert comp_lookups["apo_img"], f"Competition apo images not found under {COMPETITION_DIR}"

manifest_path = resolve_subdir(DATASET_ROOT, "manifests") / "train_apo_gray55_line.csv"
manifest = pd.read_csv(manifest_path)
stem_to_filename = dict(zip(manifest["stem"].astype(str), manifest["filename"].astype(str)))

fasc_model_path = resolve_pkl(
    [Path("/kaggle/input/notebooks/ucheozoemena/umud-train-mounted-phase-3/fasc_baseline.pkl")],
    "fasc_baseline.pkl",
)
fasc_learn = load_learner(fasc_model_path)
print("Fasc model:", fasc_model_path)

val_stems = [Path(f).stem for f in dls.valid.items]
print(f"Val images: {len(val_stems)}")

gt_rows, pred_rows = [], []
skip_manifest = skip_fasc = 0
for stem in tqdm(val_stems, desc="val umud"):
    filename = resolve_filename(stem, stem_to_filename, comp_lookups)
    if not filename:
        skip_manifest += 1
        continue
    if filename not in comp_lookups["fasc_mask"]:
        skip_fasc += 1
        continue
    img = load_gray(comp_lookups["apo_img"][filename])
    fasc_raw = load_mask(comp_lookups["fasc_mask"][filename])
    apo_raw = load_mask(comp_lookups["apo_mask"][filename])
    gt = gt_geometry_from_masks(fasc_raw, apo_raw, img.shape, MM_PER_PIXEL)
    gt_rows.append({"image_id": filename, **gt})
    pred = predict_geometry(img, fasc_learn, learn, IMG_SIZE, MM_PER_PIXEL)
    pred_rows.append(
        {
            "image_id": filename,
            "pa_deg": pred["pa_deg"],
            "fl_mm": pred["fl_mm"],
            "mt_mm": pred["mt_mm"],
            "mt_fail_reason": pred.get("mt_fail_reason"),
            "apo_cov": pred.get("apo_cov"),
        }
    )

print(f"Scored {len(pred_rows)}/{len(val_stems)} val images (skip_manifest={skip_manifest}, skip_fasc={skip_fasc})")

gt_df = pd.DataFrame(gt_rows)
pred_df = pd.DataFrame(pred_rows)
if len(pred_df) == 0:
    summary = {
        "n_total": 0,
        "n_pred_finite": 0,
        "n_gt_finite": 0,
        "n_scorable": 0,
        "val_mt_ok_pct": float("nan"),
        "val_umud_score": float("nan"),
        "val_umud_score_strict": float("nan"),
    }
    print("WARN: no dual-track val images scored — check competition mount + manifest stems")
else:
    pred_submit = pred_df[["image_id", "pa_deg", "fl_mm", "mt_mm"]]
    summary = score_summary(gt_df, pred_submit, row_id_column_name="image_id")
    print("Val UMUD summary:", summary)
    display(local_metric_report(gt_df, pred_submit))
    if pred_df["mt_mm"].isna().any():
        display(pred_df.loc[pred_df["mt_mm"].isna(), ["image_id", "mt_fail_reason", "apo_cov"]])
        display(pred_df.loc[pred_df["mt_mm"].isna(), "mt_fail_reason"].value_counts())

row = timing.iloc[0].to_dict()
for col in ("val_umud_score", "val_umud_score_strict", "val_mt_ok_pct", "n_scorable", "n_total"):
    row[col] = summary.get(col, float("nan"))
timing = pd.DataFrame([row])
timing.to_csv(WORKING / "val_umud_report.csv", index=False)
timing.to_csv(WORKING / "timing_report.csv", index=False)
display(timing)
"""
    ),
]


def write_nb(path: Path) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {path} ({len(cells)} cells)")


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks/train-apo-gray55"
    write_nb(out / "train-apo-gray55-phase-3.ipynb")
    profile = DATASET_SLUG_BY_RUN[BUILD_TRAIN_RUN]
    dataset_sources = [profile] + EXTRA_DATASET_SOURCES_BY_RUN.get(BUILD_TRAIN_RUN, [])
    meta = {
        "id": "ucheozoemena/umud-train-apo-gray55-phase-3",
        "title": "UMUD Train Apo Gray55 Phase 3",
        "code_file": "train-apo-gray55-phase-3.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": dataset_sources,
        "kernel_sources": ["ucheozoemena/umud-train-mounted-phase-3"],
        "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9",
        "machine_shape": "NvidiaTeslaT4",
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
