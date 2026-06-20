"""Timm-backed U-Net learner for fastai (adapted from walkwithfastai/wwf.vision.timm)."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from fastai.torch_core import defaults

from fastai.layers import (
    BatchNorm,
    ConvLayer,
    MergeLayer,
    ResBlock,
    SigmoidRange,
    SimpleSelfAttention,
    apply_init,
)
from fastai.vision.models.unet import ResizeToOrig
from fastai.layers import PixelShuffle_ICNR
from fastai.vision.all import (
    Adam,
    Learner,
    Module,
    SequentialEx,
    get_c,
    ifnone,
    params,
)
from fastai.vision.learner import _add_norm
from fastai.basics import store_attr


class DecoderBlock(Module):
    def __init__(
        self,
        up_in_c,
        s_in_c,
        scale=2,
        blur=False,
        final_div=True,
        act_cls=defaults.activation,
        init=nn.init.kaiming_normal_,
        norm_type=None,
        **kwargs,
    ):
        self.shuf = PixelShuffle_ICNR(
            up_in_c,
            up_in_c // 2,
            scale=scale,
            blur=blur,
            act_cls=act_cls,
            norm_type=norm_type,
            **kwargs,
        )
        self.bn = BatchNorm(s_in_c)
        self.act = act_cls()
        ni = up_in_c // 2 + s_in_c
        nf = ni if final_div else ni // 2
        self.nf = nf
        self.conv1 = ConvLayer(ni, nf, act_cls=act_cls, norm_type=norm_type, **kwargs)
        self.conv2 = ConvLayer(nf, nf, act_cls=act_cls, norm_type=norm_type, **kwargs)
        apply_init(nn.Sequential(self.shuf, self.bn, self.conv1, self.conv2), init)

    def forward(self, up_in: torch.Tensor, skip: Optional[torch.Tensor] = None):
        x = self.shuf(up_in)
        if skip is not None:
            ssh = skip.shape[-2:]
            if ssh != x.shape[-2:]:
                x = F.interpolate(x, ssh, mode="nearest")
            x = self.act(torch.cat([x, self.bn(skip)], dim=1))
        return self.conv2(self.conv1(x))


def BatchNormZero(nf, ndim=2, **kwargs):
    from fastai.layers import _get_norm

    return _get_norm("BatchNorm", nf, ndim, zero=True, **kwargs)


def _make_bottleneck(
    bottleneck,
    ni,
    act_cls=defaults.activation,
    norm_type=None,
    init=nn.init.kaiming_normal_,
    **kwargs,
):
    if bottleneck == "conv":
        seq = nn.Sequential(
            BatchNorm(ni),
            nn.ReLU(),
            ConvLayer(ni, ni * 2, act_cls=act_cls, norm_type=norm_type, **kwargs),
            ConvLayer(ni * 2, ni, act_cls=act_cls, norm_type=norm_type, **kwargs),
        )
    elif bottleneck == "attention":
        seq = nn.Sequential(
            BatchNormZero(ni),
            nn.ReLU(),
            ConvLayer(ni, ni, act_cls=act_cls, norm_type=norm_type, **kwargs),
            SimpleSelfAttention(ni, ks=1),
        )
    else:
        raise NotImplementedError(f"Bottleneck architecture {bottleneck} not implemented.")
    apply_init(seq, init)
    return seq


class UnetDecoder(Module):
    def __init__(
        self,
        encoder,
        bottleneck=None,
        blur=False,
        blur_final=True,
        norm_type=None,
        act_cls=defaults.activation,
        init=nn.init.kaiming_normal_,
        **kwargs,
    ):
        encoder_chs = encoder.feature_info.channels()[::-1]
        encoder_reds = encoder.feature_info.reduction()
        skip_channels = encoder_chs[1:]
        self.blocks = nn.ModuleList()
        up_c = encoder_chs[0]
        self.bottleneck = (
            _make_bottleneck(bottleneck, up_c, act_cls=act_cls, norm_type=norm_type, init=init, **kwargs)
            if isinstance(bottleneck, str)
            else bottleneck
        )
        for i, skip_c in enumerate(skip_channels):
            not_final = i != len(skip_channels) - 1
            do_blur = blur and (not_final or blur_final)
            scale = encoder_reds.pop() // encoder_reds[-1]
            block = DecoderBlock(
                up_c,
                skip_c,
                scale=scale,
                blur=do_blur,
                final_div=not_final,
                act_cls=act_cls,
                init=init,
                norm_type=norm_type,
                **kwargs,
            )
            up_c = block.nf
            self.blocks.append(block)
        scale = encoder_reds[0]
        self.final_shuf = (
            PixelShuffle_ICNR(up_c, scale=scale, act_cls=act_cls, norm_type=norm_type, **kwargs)
            if scale != 1
            else None
        )
        self.nf = up_c

    def forward(self, x: List[torch.Tensor]):
        x.reverse()
        skips = x[1:]
        x = x[0]
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, skip)
        if self.final_shuf is not None:
            x = self.final_shuf(x)
        return x


class UnetHead(SequentialEx):
    def __init__(
        self,
        up_c,
        n_out,
        last_cross=False,
        n_in=None,
        bottle=False,
        norm_type=None,
        act_cls=defaults.activation,
        init=nn.init.kaiming_normal_,
        y_range=None,
        **kwargs,
    ):
        layers = nn.ModuleList([ResizeToOrig()])
        if last_cross:
            if n_in is None:
                raise AttributeError("You must specify `n_in` if `last_cross=True`.")
            up_c += n_in
            layers.extend(
                [
                    MergeLayer(dense=True),
                    ResBlock(
                        1,
                        up_c,
                        up_c // 2 if bottle else up_c,
                        act_cls=act_cls,
                        norm_type=norm_type,
                        **kwargs,
                    ),
                ]
            )
        layers.append(nn.Conv2d(up_c, n_out, kernel_size=(1, 1)))
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)
        apply_init(self, init)


class TimmUnet(SequentialEx):
    def __init__(
        self,
        encoder="resnet50",
        n_in=3,
        n_out=1,
        encoder_kwargs=None,
        encoder_indices=None,
        blur=False,
        blur_final=True,
        bottleneck=None,
        last_cross=True,
        norm_type=None,
        act_cls=defaults.activation,
        init=nn.init.kaiming_normal_,
        pretrained=True,
        y_range=None,
        bottle=False,
        **kwargs,
    ):
        encoder_kwargs = encoder_kwargs or {}
        self.encoder = create_model(
            encoder,
            features_only=True,
            out_indices=encoder_indices,
            in_chans=n_in,
            pretrained=pretrained,
            **encoder_kwargs,
        )
        self.decoder = UnetDecoder(
            self.encoder,
            bottleneck=bottleneck,
            blur=blur,
            blur_final=blur_final,
            norm_type=norm_type,
            act_cls=act_cls,
            init=init,
            **kwargs,
        )
        self.head = UnetHead(
            self.decoder.nf,
            n_out,
            last_cross=last_cross,
            n_in=n_in,
            bottle=bottle,
            norm_type=norm_type,
            act_cls=act_cls,
            init=init,
            y_range=y_range,
            **kwargs,
        )
        super().__init__(nn.Sequential(self.encoder, self.decoder), *self.head)

    @property
    def default_cfg(self):
        return self.encoder.default_cfg

    @property
    def feature_info(self):
        return self.encoder.feature_info


def _get_params_from_attrs(m, ls):
    from fastai.torch_core import getattrs

    return params(nn.Sequential(*getattrs(m, *ls)))


def _get_params_from_modules(modules):
    return params(nn.Sequential(*modules))


def split_nested_list(l, idxs, left_in=None, right_in=None):
    from fastai.data.core import L

    left_in, right_in = ifnone(left_in, L()), ifnone(right_in, L())
    idx = int(idxs[0])
    if len(idxs) == 1:
        left, right = l[:idx], l[idx:]
    else:
        left, right = split_nested_list(l[idx], idxs[1:], l[:idx], l[idx + 1 :])
    return L(L(*left_in, *left), L(*right, *right_in))


def _timm_splitter(m):
    from fastai.torch_core import getattrs
    from fastai.data.core import L

    split_path = m.feature_info.module_name(0).split(".")

    def module_children(mod):
        return getattrs(mod, *list(mod._modules.keys()))

    enc = m.encoder
    enc_names = list(enc._modules.keys())
    enc_mods = module_children(enc)

    # LeViT-style single wrapper
    if len(enc_mods) == 1 and enc_mods[0]._modules:
        inner = enc_mods[0]
        inner_names = list(inner._modules.keys())
        inner_mods = module_children(inner)
        cut = _split_cut_index(inner_names, split_path)
        encoder_early = _get_params_from_modules(inner_mods[:cut])
        encoder_late = _get_params_from_modules(inner_mods[cut:])
    else:
        cut = _split_cut_index(enc_names, split_path)
        encoder_early = _get_params_from_modules(enc_mods[:cut])
        encoder_late = _get_params_from_modules(enc_mods[cut:])

    decoder = _get_params_from_attrs(m.decoder, L(*list(m.decoder._modules.keys())))
    head = _get_params_from_attrs(m.head, L(*list(m.head._modules.keys())))
    return L(encoder_early, encoder_late, L(*decoder, *head))


def _split_cut_index(names: list[str], split_path: list[str]) -> int:
    """Index in `names` where fine-tuning should begin (modules before are frozen)."""
    if not split_path:
        return max(1, len(names) // 2)
    head = split_path[0]
    if head in names:
        return names.index(head)
    if len(split_path) >= 2 and split_path[1].isdigit():
        flat = f"{head}_{split_path[1]}"
        if flat in names:
            return names.index(flat)
    flat = "_".join(split_path[:2]) if len(split_path) >= 2 else head
    if flat in names:
        return names.index(flat)
    for i, n in enumerate(names):
        if n.startswith(head):
            return i
    return max(1, len(names) // 2)


def _timm_stats(m):
    return tuple((list(m.default_cfg[stat]) for stat in ("mean", "std")))


def timm_unet_learner(
    dls,
    arch,
    normalize=True,
    n_in=None,
    n_out=None,
    pretrained=True,
    loss_func=None,
    opt_func=Adam,
    lr=defaults.lr,
    splitter=None,
    cbs=None,
    metrics=None,
    path=None,
    model_dir="models",
    wd=None,
    wd_bn_bias=False,
    train_bn=True,
    moms=(0.95, 0.85, 0.95),
    bottleneck="conv",
    **kwargs,
):
    """Build a U-Net learner with a timm encoder."""
    n_in = ifnone(n_in, dls.one_batch()[0].shape[1])
    n_out = ifnone(n_out, get_c(dls))
    assert n_out, "`n_out` is not defined — set `dls.c` or pass `n_out`"
    model = TimmUnet(arch, n_in=n_in, n_out=n_out, pretrained=pretrained, bottleneck=bottleneck, **kwargs)
    if normalize:
        _add_norm(dls, {"stats": _timm_stats(model)}, pretrained)
    splitter = ifnone(splitter, _timm_splitter)
    learn = Learner(
        dls=dls,
        model=model,
        loss_func=loss_func,
        opt_func=opt_func,
        lr=lr,
        splitter=splitter,
        cbs=cbs,
        metrics=metrics,
        path=path,
        model_dir=model_dir,
        wd=wd,
        wd_bn_bias=wd_bn_bias,
        train_bn=train_bn,
        moms=moms,
    )
    if pretrained:
        learn.freeze()
    store_attr("arch,normalize,n_out,pretrained", self=learn, **kwargs)
    return learn
