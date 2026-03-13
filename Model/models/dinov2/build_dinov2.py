import copy
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..ImageEncoder.vit.peft_modules import BoundaryAwareGatedFusionBlock, CPIABlock


def build_dinov2_vits14(args, checkpoint=None):
    return DinoV2DualEncoder(args, checkpoint=checkpoint, model_name="dinov2_vits14")


def build_dinov2_vitb14(args, checkpoint=None):
    return DinoV2DualEncoder(args, checkpoint=checkpoint, model_name="dinov2_vitb14")


def build_dinov2_vitl14(args, checkpoint=None):
    return DinoV2DualEncoder(args, checkpoint=checkpoint, model_name="dinov2_vitl14")


def build_dinov2_vitg14(args, checkpoint=None):
    return DinoV2DualEncoder(args, checkpoint=checkpoint, model_name="dinov2_vitg14")


def load_dinov2_weights_full(model, checkpoint, strict=True):
    ckpt_path = _resolve_checkpoint_path(checkpoint)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"DINOv2 checkpoint not found: {checkpoint}")

    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(state)
    state_dict = _strip_prefixes(state_dict)
    state_dict, dropped_unexpected, dropped_mismatch = _filter_state_dict(state_dict, model.state_dict())

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if strict and (missing or unexpected or dropped_unexpected or dropped_mismatch):
        raise RuntimeError(
            "[DINOv2] Strict load failed. "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"dropped_unexpected={dropped_unexpected} dropped_mismatch={dropped_mismatch}"
        )
    print(
        f"[DINOv2] Loaded weights from {ckpt_path}. "
        f"missing={len(missing)}, unexpected={len(unexpected)}, "
        f"dropped_unexpected={dropped_unexpected}, dropped_mismatch={dropped_mismatch}, strict={strict}"
    )


def _hub_load(repo_or_dir: str, model_name: str, source: Optional[str] = None):
    kwargs = {"pretrained": False}
    if source is not None:
        kwargs["source"] = source
    try:
        return torch.hub.load(repo_or_dir, model_name, trust_repo=True, **kwargs)
    except TypeError:
        return torch.hub.load(repo_or_dir, model_name, **kwargs)


class DinoV2DualEncoder(nn.Module):
    def __init__(self, args, checkpoint=None, model_name="dinov2_vitl14"):
        super().__init__()
        hub_dir = getattr(args, "dinov2_hub_dir", None)
        if hub_dir is None and checkpoint is not None:
            hub_dir = _infer_hub_dir_from_checkpoint(checkpoint)
        if not hub_dir:
            raise RuntimeError(
                "[DINOv2] Local dinov2 repo not set. "
                "Put the repo in dinov2_hub/ or pass -dinov2_hub_dir."
            )
        self.backbone = _hub_load(hub_dir, model_name, source="local")

        strict = getattr(args, "dinov2_strict", False)
        if checkpoint:
            load_dinov2_weights_full(self.backbone, checkpoint, strict=strict)

        self.embed_dim = _infer_embed_dim(self.backbone)
        self.patch_size = _infer_patch_size(self.backbone)
        self.grid_size = _infer_grid_size(args, self.patch_size)
        self.backbone_blocks = _flatten_backbone_blocks(self.backbone)
        self.tap_indices = _resolve_tap_indices(args, model_name, len(self.backbone_blocks))
        self.use_extra_patch_embed = getattr(args, "dinov2_use_extra_patch_embed", True)
        self.cpia_enabled = getattr(args, "use_cpia", True)
        self.dgfm_enabled = getattr(args, "use_dgfm", True)

        self.extra_patch_embed = None
        if self.use_extra_patch_embed:
            if not hasattr(self.backbone, "patch_embed"):
                raise AttributeError("[DINOv2] backbone has no patch_embed; cannot build extra_patch_embed.")
            self.extra_patch_embed = copy.deepcopy(self.backbone.patch_embed)
            self.extra_patch_embed.load_state_dict(self.backbone.patch_embed.state_dict(), strict=True)

        peft_ratio = getattr(args, "peft_ratio", 0.25)
        self.peft_cpia_blocks = nn.ModuleList(
            [CPIABlock(self.embed_dim, ratio=peft_ratio) for _ in self.tap_indices]
        )
        self.peft_dgfm_blocks = nn.ModuleList(
            [
                BoundaryAwareGatedFusionBlock(
                    dims=self.embed_dim,
                    ratio=peft_ratio,
                )
                for _ in range(len(self.tap_indices))
            ]
        )
        self.peft_fuse_norms = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in self.tap_indices]
        )
        self.dual_proj_layers = nn.ModuleList(
            [nn.Conv2d(self.embed_dim, 256, kernel_size=1, bias=False) for _ in self.tap_indices]
        )

        for param in self.backbone.parameters():
            param.requires_grad = False
        if self.extra_patch_embed is not None:
            for param in self.extra_patch_embed.parameters():
                param.requires_grad = True

    def _prepare_tokens_with_patch_embed(
        self, x: torch.Tensor, patch_embed: nn.Module, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, _, w, h = x.shape
        x = patch_embed(x)
        if masks is not None:
            if not hasattr(self.backbone, "mask_token"):
                raise AttributeError("[DINOv2] backbone has no mask_token for masked forward.")
            x = torch.where(masks.unsqueeze(-1), self.backbone.mask_token.to(x.dtype).unsqueeze(0), x)

        if not hasattr(self.backbone, "cls_token"):
            raise AttributeError("[DINOv2] backbone has no cls_token for token preparation.")
        x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if hasattr(self.backbone, "interpolate_pos_encoding"):
            x = x + self.backbone.interpolate_pos_encoding(x, w, h)
        elif hasattr(self.backbone, "pos_embed"):
            x = x + self.backbone.pos_embed
        else:
            raise AttributeError("[DINOv2] backbone has no positional embedding interface.")

        if getattr(self.backbone, "register_tokens", None) is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.backbone.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x

    def _forward_features_with_patch_embed(
        self, x: torch.Tensor, patch_embed: nn.Module, masks: Optional[torch.Tensor] = None
    ):
        if not hasattr(self.backbone, "blocks") or not hasattr(self.backbone, "norm"):
            raise AttributeError("[DINOv2] backbone does not expose blocks/norm for shared-transformer forward.")

        x = self._prepare_tokens_with_patch_embed(x, patch_embed, masks=masks)
        for blk in self.backbone.blocks:
            x = blk(x)

        x_norm = self.backbone.norm(x)
        num_register_tokens = int(getattr(self.backbone, "num_register_tokens", 0))
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _infer_hw(self, x: torch.Tensor, tokens: torch.Tensor) -> Tuple[int, int]:
        if self.patch_size is not None:
            h = x.shape[-2] // self.patch_size
            w = x.shape[-1] // self.patch_size
            return h, w
        n = tokens.shape[1]
        h = int(math.sqrt(n))
        w = n // h
        return h, w

    def _split_special_tokens(self, tokens: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        expected = h * w
        if tokens.shape[1] < expected:
            raise ValueError("Patch token count is smaller than expected.")
        special_len = tokens.shape[1] - expected
        return tokens[:, :special_len, :], tokens[:, -expected:, :]

    def _tokens_to_feature_map(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, _, c = tokens.shape
        return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_seq = self._prepare_tokens_with_patch_embed(x, self.backbone.patch_embed)
        y_patch_embed = self.extra_patch_embed if self.extra_patch_embed is not None else self.backbone.patch_embed
        y_seq = self._prepare_tokens_with_patch_embed(y, y_patch_embed)

        h, w = self._infer_hw(x, x_seq)
        expected = h * w
        stage_fused_feats: List[torch.Tensor] = []
        stage_rgb_feats: List[torch.Tensor] = []
        stage_dsm_feats: List[torch.Tensor] = []

        block_start = 0
        for stage_idx, tap_idx in enumerate(self.tap_indices):
            x_special, x_patch = self._split_special_tokens(x_seq, h, w)
            y_special, y_patch = self._split_special_tokens(y_seq, h, w)

            if self.cpia_enabled:
                x_cpia, y_cpia = self.peft_cpia_blocks[stage_idx](x_patch, y_patch, h, w)
                x_patch = x_patch + x_cpia
                y_patch = y_patch + y_cpia
            x_seq = torch.cat((x_special, x_patch), dim=1)
            y_seq = torch.cat((y_special, y_patch), dim=1)

            for blk_idx in range(block_start, tap_idx + 1):
                blk = self.backbone_blocks[blk_idx]
                x_seq = blk(x_seq)
                y_seq = blk(y_seq)
            block_start = tap_idx + 1

            x_stage = self.backbone.norm(x_seq)[:, -expected:, :]
            y_stage = self.backbone.norm(y_seq)[:, -expected:, :]
            if self.dgfm_enabled:
                fused_stage = self.peft_dgfm_blocks[stage_idx](x_stage, y_stage, h, w)
                fused_stage = self.peft_fuse_norms[stage_idx](fused_stage)
            else:
                fused_stage = 0.5 * (x_stage + y_stage)

            proj = self.dual_proj_layers[stage_idx]
            x_stage_feat = proj(self._tokens_to_feature_map(x_stage, h, w))
            y_stage_feat = proj(self._tokens_to_feature_map(y_stage, h, w))
            fused_stage_feat = proj(self._tokens_to_feature_map(fused_stage, h, w))

            stage_rgb_feats.append(x_stage_feat)
            stage_dsm_feats.append(y_stage_feat)
            stage_fused_feats.append(fused_stage_feat)

        return stage_fused_feats, stage_rgb_feats, stage_dsm_feats


def _resolve_checkpoint_path(path):
    if path is None:
        return None
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        for ext in (".pth", ".pt", ".bin"):
            files = sorted(path.glob(f"*{ext}"))
            if files:
                return files[0]
        files = sorted([p for p in path.iterdir() if p.is_file()])
        if files:
            return files[0]
    return path


def _infer_hub_dir_from_checkpoint(checkpoint) -> Optional[Path]:
    ckpt_path = _resolve_checkpoint_path(checkpoint)
    if ckpt_path is None:
        return None
    for parent in [ckpt_path.parent] + list(ckpt_path.parents):
        if (parent / "hubconf.py").is_file():
            return parent
    return None


def _extract_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model", "teacher", "student"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _strip_prefixes(state_dict):
    prefixes = (
        "teacher.backbone.",
        "student.backbone.",
        "model.backbone.",
        "backbone.",
        "module.",
        "teacher.",
        "student.",
        "model.",
    )
    stripped = state_dict
    for prefix in prefixes:
        if any(k.startswith(prefix) for k in stripped.keys()):
            stripped = {
                k[len(prefix):] if k.startswith(prefix) else k: v for k, v in stripped.items()
            }
    return stripped


def _filter_state_dict(state_dict, model_state_dict):
    filtered = {}
    dropped_unexpected = 0
    dropped_mismatch = 0
    for key, value in state_dict.items():
        if key not in model_state_dict:
            dropped_unexpected += 1
            continue
        if model_state_dict[key].shape != value.shape:
            dropped_mismatch += 1
            continue
        filtered[key] = value
    return filtered, dropped_unexpected, dropped_mismatch


def _infer_embed_dim(backbone: nn.Module) -> int:
    for attr in ("embed_dim", "num_features", "hidden_dim"):
        if hasattr(backbone, attr):
            return int(getattr(backbone, attr))
    if hasattr(backbone, "patch_embed") and hasattr(backbone.patch_embed, "proj"):
        return int(backbone.patch_embed.proj.out_channels)
    raise ValueError("Unable to infer DINOv2 embed dim.")


def _infer_patch_size(backbone: nn.Module) -> Optional[int]:
    if hasattr(backbone, "patch_size"):
        val = getattr(backbone, "patch_size")
        if isinstance(val, (tuple, list)):
            return int(val[0])
        return int(val)
    if hasattr(backbone, "patch_embed"):
        patch_size = getattr(backbone.patch_embed, "patch_size", None)
        if isinstance(patch_size, (tuple, list)):
            return int(patch_size[0])
        if patch_size is not None:
            return int(patch_size)
        if hasattr(backbone.patch_embed, "proj"):
            ks = backbone.patch_embed.proj.kernel_size
            if isinstance(ks, (tuple, list)):
                return int(ks[0])
            return int(ks)
    return None


def _infer_grid_size(args, patch_size: Optional[int]) -> Optional[Tuple[int, int]]:
    if patch_size is None:
        return None
    img_size = getattr(args, "image_size", None)
    if img_size is None:
        return None
    return (int(img_size // patch_size), int(img_size // patch_size))


def _flatten_backbone_blocks(backbone: nn.Module) -> List[nn.Module]:
    if not hasattr(backbone, "blocks"):
        raise AttributeError("[DINOv2] backbone does not expose transformer blocks.")
    flat_blocks: List[nn.Module] = []
    for blk in backbone.blocks:
        if isinstance(blk, nn.ModuleList):
            for sub_blk in blk:
                if isinstance(sub_blk, nn.Identity):
                    continue
                flat_blocks.append(sub_blk)
            continue
        if isinstance(blk, nn.Identity):
            continue
        flat_blocks.append(blk)
    if not flat_blocks:
        raise ValueError("[DINOv2] No valid transformer blocks found for tap extraction.")
    return flat_blocks


def _resolve_tap_indices(args, model_name: str, total_blocks: int) -> List[int]:
    raw_indices = getattr(args, "dinov2_tap_indices", None)
    parsed = _parse_tap_indices(raw_indices)
    if parsed:
        taps = _sanitize_tap_indices(parsed, total_blocks)
    else:
        taps = _sanitize_tap_indices(_default_tap_indices(model_name, total_blocks), total_blocks)
    if len(taps) != 4:
        taps = _sanitize_tap_indices(_uniform_tap_indices(total_blocks, 4), total_blocks)
    return taps


def _parse_tap_indices(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        out = []
        for v in value:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        out = []
        for chunk in text.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                out.append(int(chunk))
            except ValueError:
                continue
        return out
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def _default_tap_indices(model_name: str, total_blocks: int) -> List[int]:
    name = model_name.lower()
    if "vitg14" in name:
        return [9, 19, 29, 39]
    if "vitl14" in name:
        return [5, 11, 17, 23]
    if "vitb14" in name or "vits14" in name:
        return [2, 5, 8, 11]
    return _uniform_tap_indices(total_blocks, 4)


def _uniform_tap_indices(total_blocks: int, n_taps: int) -> List[int]:
    if total_blocks <= 0:
        return []
    return [max(0, min(total_blocks - 1, (i + 1) * total_blocks // n_taps - 1)) for i in range(n_taps)]


def _sanitize_tap_indices(indices: Sequence[int], total_blocks: int) -> List[int]:
    valid = [int(idx) for idx in indices if 0 <= int(idx) < total_blocks]
    valid = sorted(set(valid))
    return valid
