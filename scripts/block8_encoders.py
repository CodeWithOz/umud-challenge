"""Block 8 encoder sweep — one Kaggle notebook per architecture (200×5ep)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Status = Literal["pending", "train_complete", "test_complete", "submitted", "rejected"]


@dataclass(frozen=True)
class EncoderSpec:
    slug: str
    family: str
    arch: str
    export_name: str
    params_m: float
    note: str
    status: Status = "pending"
    val_dice: float | None = None
    val_umud: float | None = None
    val_mt_ok_pct: float | None = None
    test_mt_ok_pct: float | None = None
    lb_score: float | None = None
    img_size: int = 256

    @property
    def kernel_id(self) -> str:
        return f"ucheozoemena/umud-train-encoder-{self.slug}-phase-3"

    @property
    def notebook_dir(self) -> str:
        return f"notebooks/train-encoder-{self.slug}"

    @property
    def ipynb_name(self) -> str:
        return f"train-encoder-{self.slug}-phase-3.ipynb"


DATASET_SLUG = "ucheozoemena/umud-aligned-apo-gray55-line-timing-200"
EPOCHS = 5

# Smallest / entry variant per family — run in this order
BLOCK8_ENCODERS: tuple[EncoderSpec, ...] = (
    EncoderSpec(
        slug="levit128s",
        family="LeViT",
        arch="levit_128s",
        export_name="apo_gray55_line_200_levit128s.pkl",
        params_m=9.2,
        note="LeViT-128S — light hybrid ViT/CNN; IMG_SIZE 224 (256 breaks U-Net skips)",
        img_size=224,
    ),
    EncoderSpec(
        slug="resnetv2-18",
        family="ResNet V2",
        arch="resnetv2_18",
        export_name="apo_gray55_line_200_rv2_18.pkl",
        params_m=11.7,
        note="ResNetV2-18 — pre-activation ResNet",
    ),
    EncoderSpec(
        slug="convnextv2-atto",
        family="ConvNeXt V2",
        arch="convnextv2_atto",
        export_name="apo_gray55_line_200_cnxv2_atto.pkl",
        params_m=3.7,
        note="ConvNeXt V2 Atto — smallest V2",
    ),
    EncoderSpec(
        slug="efficientnetv2-rw-t",
        family="EfficientNet V2",
        arch="efficientnetv2_rw_t",
        export_name="apo_gray55_line_200_env2_rw_t.pkl",
        params_m=13.0,
        note="EfficientNetV2-RW-T — tiny RW variant",
    ),
    EncoderSpec(
        slug="maxvit-nano",
        family="MaxViT",
        arch="maxvit_nano_rw_256",
        export_name="apo_gray55_line_200_maxvit_nano.pkl",
        params_m=31.0,
        note="MaxViT Nano RW @256",
    ),
    EncoderSpec(
        slug="maxxvitv2-nano",
        family="MaxViT V2",
        arch="maxxvitv2_nano_rw_256",
        export_name="apo_gray55_line_200_maxxvitv2_nano.pkl",
        params_m=23.0,
        note="MaxXViT V2 Nano RW @256 (timm maxxvitv2)",
    ),
    EncoderSpec(
        slug="maxvit-tiny",
        family="MaxViT",
        arch="maxvit_rmlp_tiny_rw_256",
        export_name="apo_gray55_line_200_maxvit_tiny.pkl",
        params_m=28.6,
        note="MaxViT RMLP Tiny RW @256 — next pretrained tier above nano (Block 11)",
    ),
)


def get_encoder(slug: str) -> EncoderSpec:
    for enc in BLOCK8_ENCODERS:
        if enc.slug == slug:
            return enc
    raise KeyError(f"Unknown Block 8 encoder slug: {slug!r}")
