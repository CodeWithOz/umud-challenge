"""Block 10 — re-score Block 7/8 encoder architectures with Block 9 calibration.

Uses existing submission_debug.csv pixel geometry (no retrain). Applies the
production s2 calibration (PA=18, FL/MT split scales + MT shrink + NaN fallback)
and ranks models with the validated tracking metric (data/fit_mu.npy).

Run: .venv/bin/python scripts/block10_eval.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TOL = {"pa": 6.0, "fl": 12.0, "mt": 3.0}

# Block 9 s2 production calibration (scripts/build_submission_nb.py)
S2_PA_TARGET = 18.0
S2_MU_FL = 74.48
S2_MU_MT = 21.5
S2_S_FL = 0.088
S2_S_MT = 0.0725
S2_MT_SHRINK = 0.45

# Legacy Block 7/8 evaluation (uniform MM=0.075, raw PA, no NaN fallback)
LEGACY_MM = 0.075

c0, MU_PA, MU_FL, MU_MT = np.load(ROOT / "data/fit_mu.npy")


@dataclass(frozen=True)
class ArchSpec:
    block: str
    name: str
    debug_csv: str
    lb_legacy: float | None  # public LB @ MM=0.075 (None if not submitted)
    note: str = ""


# All Block 7/8 architectures with saved test debug geometry.
ARCHITECTURES: tuple[ArchSpec, ...] = (
    # Block 8
    ArchSpec("8", "resnetv2_18", "data/kaggle-outputs/block8/resnetv2-18/submit/submission_debug.csv", 1.84197),
    ArchSpec("8", "resnetv2_18_s2", "data/kaggle-outputs/block10/rv2-s2/submission_debug.csv", None, "Block 10 rv2 + s2 cal submit v33"),
    ArchSpec("8", "maxvit_nano", "data/kaggle-outputs/block8/maxvit-nano/submit/submission_debug.csv", 1.82151),
    ArchSpec("8", "levit128s", "data/kaggle-outputs/block8/levit128s/submit/submission_debug.csv", 1.91255),
    ArchSpec("8", "efficientnetv2_rw_t", "data/kaggle-outputs/block8/efficientnetv2-rw-t/submit/submission_debug.csv", 1.98186),
    ArchSpec("8", "convnextv2_atto", "data/kaggle-outputs/block8/convnextv2-atto/submit/submission_debug.csv", None, "94% mt_ok — not LB submitted"),
    ArchSpec("8", "maxxvitv2_nano", "data/kaggle-outputs/block8/maxxvitv2-nano/submit/submission_debug.csv", None, "99.7% mt_ok — not LB submitted"),
    # Block 7
    ArchSpec("7", "resnet18", "data/kaggle-outputs/block7-test-eval/run13-resnet18/submission_debug.csv", 1.86662),
    ArchSpec("7", "regnetx_004", "data/kaggle-outputs/block7-test-eval/run19-regnetx_004/submission_debug.csv", 1.87201),
    ArchSpec("7", "efficientnet_b1", "data/kaggle-outputs/block7-test-eval/run17-efficientnet_b1/submission_debug.csv", 1.88316),
    ArchSpec("7", "mobilenetv3_small_100", "data/kaggle-outputs/block7-test-eval/run18-mobilenetv3_small_100/submission_debug.csv", 1.91682),
    ArchSpec("7", "efficientnet_b0", "data/kaggle-outputs/block7-test-eval/run16-efficientnet_b0/submission_debug.csv", None, "74% mt_ok — rejected"),
    ArchSpec("7", "convnext_tiny", "data/kaggle-outputs/block7-cxt-submit/submission_debug.csv", None, "71% mt_ok — rejected"),
    # Block 10 debug runs (notebook only — no LB submit)
    ArchSpec("6", "resnet50", "data/kaggle-outputs/block10/r50-debug/submission_debug.csv", 1.87312, "Block 10 debug"),
    ArchSpec("6", "resnet34", "data/kaggle-outputs/block10/r34-debug/submission_debug.csv", 1.91296, "Block 10 debug"),
    ArchSpec("7", "convnext_small", "data/kaggle-outputs/block10/cxs-debug/submission_debug.csv", None, "Block 10 debug — first train"),
)


def track_score(pa: np.ndarray, fl: np.ndarray, mt: np.ndarray) -> tuple[float, float, float, float]:
    g_pa = np.mean(np.abs(pa - MU_PA)) / TOL["pa"]
    g_fl = np.mean(np.abs(fl - MU_FL)) / TOL["fl"]
    g_mt = np.mean(np.abs(mt - MU_MT)) / TOL["mt"]
    return c0 + (g_pa + g_fl + g_mt) / 3.0, g_pa / 3, g_fl / 3, g_mt / 3


def legacy_calibrate(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pa = df["pa_deg"].to_numpy(float)
    fl = df["fl_px"].to_numpy(float) * LEGACY_MM
    mt = df["mt_px"].to_numpy(float) * LEGACY_MM
    return pa, fl, mt


def s2_calibrate(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flpx = df["fl_px"].to_numpy(float)
    mtpx = df["mt_px"].to_numpy(float)
    fl = np.where(np.isfinite(flpx), flpx * S2_S_FL, S2_MU_FL)
    mt = np.where(
        np.isfinite(mtpx),
        S2_MU_MT + S2_MT_SHRINK * (mtpx * S2_S_MT - S2_MU_MT),
        S2_MU_MT,
    )
    pa = np.full(len(df), S2_PA_TARGET)
    return pa, fl, mt


def mt_ok_raw(df: pd.DataFrame) -> tuple[int, int, float]:
    mtpx = df["mt_px"].to_numpy(float)
    ok = int(np.isfinite(mtpx).sum())
    n = len(df)
    return ok, n - ok, 100.0 * ok / n


def eval_arch(spec: ArchSpec) -> dict:
    path = ROOT / spec.debug_csv
    if not path.exists():
        return {"name": spec.name, "block": spec.block, "status": "missing_csv", "path": str(path)}

    df = pd.read_csv(path)
    mt_ok, mt_nan, mt_pct = mt_ok_raw(df)

    pa_l, fl_l, mt_l = legacy_calibrate(df)
    pa_s, fl_s, mt_s = s2_calibrate(df)

    track_l, _, _, _ = track_score(pa_l, fl_l, mt_l)
    track_s, gpa, gfl, gmt = track_score(pa_s, fl_s, mt_s)

    # Anchor: maxvit s2 actual public LB vs its tracking score (understates floor).
    lb_estimate = None
    if np.isfinite(track_s):
        lb_estimate = round(track_s + (1.06757 - 0.65379), 5)

    nan_s = int(np.sum(~np.isfinite(pa_s)) + np.sum(~np.isfinite(fl_s)) + np.sum(~np.isfinite(mt_s)))

    return {
        "block": spec.block,
        "name": spec.name,
        "status": "ok",
        "n": len(df),
        "mt_ok_raw": mt_ok,
        "mt_nan_raw": mt_nan,
        "mt_ok_pct": round(mt_pct, 2),
        "lb_legacy": spec.lb_legacy,
        "track_legacy": round(track_l, 5),
        "track_s2": round(track_s, 5),
        "lb_estimate_s2": lb_estimate,
        "track_delta": round(track_s - track_l, 5),
        "contrib_pa": round(gpa, 4),
        "contrib_fl": round(gfl, 4),
        "contrib_mt": round(gmt, 4),
        "nan_after_s2": nan_s,
        "mt_median_s2": round(float(np.median(mt_s)), 2),
        "mt_std_s2": round(float(np.std(mt_s)), 2),
        "note": spec.note,
    }


def main():
    rows = [eval_arch(spec) for spec in ARCHITECTURES]
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    ok_rows.sort(key=lambda r: r["track_s2"])

    print("=" * 100)
    print("BLOCK 10 — encoder re-rank with Block 9 s2 calibration + tracking metric")
    print("=" * 100)
    print(f"Tracking mu: PA={MU_PA:.2f}  FL={MU_FL:.2f}  MT={MU_MT:.2f}  c0={c0:.4f}")
    print(f"s2 cal: PA={S2_PA_TARGET}  FL scale={S2_S_FL}  MT scale={S2_S_MT}  "
          f"MU_FL={S2_MU_FL}  MU_MT={S2_MU_MT}  shrink={S2_MT_SHRINK}")
    print("No retrain — pixel geometry from saved submission_debug.csv files.\n")

    hdr = (f"{'rank':>4} {'block':>5} {'encoder':<22} {'mt_ok%':>6} {'LB@075':>8} "
           f"{'LB~s2':>8} {'track_s2':>9}  note")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(ok_rows, 1):
        lb = f"{r['lb_legacy']:.5f}" if r["lb_legacy"] is not None else "   —"
        lb_s2 = f"{r['lb_estimate_s2']:.5f}" if r.get("lb_estimate_s2") is not None else "   —"
        note = r["note"][:28] if r["note"] else ""
        print(
            f"{i:4d} {r['block']:>5} {r['name']:<22} {r['mt_ok_pct']:6.1f} {lb:>8} "
            f"{lb_s2:>8} {r['track_s2']:9.5f}  {note}"
        )

    missing = [r for r in rows if r.get("status") != "ok"]
    if missing:
        print("\nMissing debug CSV (need Kaggle submission re-run):")
        for r in missing:
            print(f"  - {r['name']}: {r.get('path', r.get('status'))}")

    print("\n--- MT term decomposition (s2 tracking contrib) ---")
    for r in ok_rows[:5]:
        print(f"  {r['name']:<22} PA={r['contrib_pa']:.3f} FL={r['contrib_fl']:.3f} "
              f"MT={r['contrib_mt']:.3f}  MT_std={r['mt_std_s2']:.2f}")

    print(f"\nLB~s2 = track_s2 + (1.06757 - 0.65379) anchored on maxvit s2 public submit.")
    best = ok_rows[0]
    print(f"\nBest encoder (tracking s2): {best['name']} -> {best['track_s2']:.5f}")
    if best["name"] != "maxvit_nano":
        mv = next(r for r in ok_rows if r["name"] == "maxvit_nano")
        print(f"  vs maxvit_nano: {mv['track_s2']:.5f} (delta {best['track_s2'] - mv['track_s2']:+.5f})")

    out_dir = ROOT / "data/kaggle-outputs/block10"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "block10_results.json"
    out_csv = out_dir / "block10_results.csv"
    with open(out_json, "w") as f:
        json.dump(ok_rows, f, indent=2)
    pd.DataFrame(ok_rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_json.relative_to(ROOT)} and {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
