import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.0625, act_layer=nn.ReLU, skip_connect=True, prompt_add=False):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.D_fc1.weight, a=np.sqrt(5))
            nn.init.zeros_(self.D_fc2.weight)
            nn.init.zeros_(self.D_fc1.bias)
            nn.init.zeros_(self.D_fc2.bias)
        self.prompt_add = prompt_add
        if prompt_add:
            self.D_fc_prompt = nn.Linear(D_features, D_hidden_features)

    def forward(self, x, prompt=None):
        xs = self.D_fc1(x)
        if self.prompt_add and prompt is not None:
            prompts = self.D_fc_prompt(prompt)
            xs = xs + prompts
        xs = self.act(xs)
        xs = nn.functional.dropout(xs, p=0.1, training=self.training)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


def init_tfts(dim):
    gamma = nn.Parameter(torch.ones(dim))
    beta = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(gamma, mean=1, std=.02)
    nn.init.normal_(beta, std=.02)
    return gamma, beta


def apply_tfts(x, gamma, beta):
    assert gamma.shape == beta.shape
    if x.shape[-1] == gamma.shape[0]:
        return x * gamma + beta
    if x.shape[1] == gamma.shape[0]:
        return x * gamma.view(1, -1, 1) + beta.view(1, -1, 1)
    raise ValueError('the input tensor shape does not match the shape of the scale factor.')


class CPIABlock(nn.Module):
    def __init__(self, dim, ratio):
        super().__init__()
        self.D_fc1 = nn.Linear(dim, int(dim * ratio))
        self.D_fc2 = nn.Linear(dim, int(dim * ratio))
        self.P_fc2 = nn.Linear(int(dim * ratio * 2), int(dim * ratio))
        self.U_fc1 = nn.Linear(int(dim * ratio), dim)
        self.act = nn.GELU()
        self.tfts_gamma_rgb, self.tfts_beta_rgb = init_tfts(dim)
        self.tfts_gamma_dsm, self.tfts_beta_dsm = init_tfts(dim)

    def forward(self, x_rgb, x_dsm, H, W):
        # Unifying dual-modality features.
        x_rgb = self.D_fc1(x_rgb)
        x_dsm = self.D_fc2(x_dsm)
        x = torch.cat([x_rgb, x_dsm], dim=-1)
        x = self.P_fc2(x)
        x = self.U_fc1(x)

        # Prompt generation.
        p_rgb = apply_tfts(x, self.tfts_gamma_rgb, self.tfts_beta_rgb)
        p_dsm = apply_tfts(x, self.tfts_gamma_dsm, self.tfts_beta_dsm)

        return x + p_rgb, x + p_dsm


class BoundaryAwareGatedFusionBlock(nn.Module):
    def __init__(self, dims, ratio=0.25):
        super().__init__()
        hidden_dim = max(32, int(dims * min(ratio, 0.125)))

        self.rgb_reduce = nn.Conv2d(dims, hidden_dim, kernel_size=1, bias=False)
        self.dsm_reduce = nn.Conv2d(dims, hidden_dim, kernel_size=1, bias=False)

        self.edge_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )
        self.gate_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 5, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dims, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dims, kernel_size=1, bias=False),
        )
        self.refine_scale = nn.Parameter(torch.tensor(0.1))

        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        nn.init.zeros_(self.gate_net[-2].weight)
        nn.init.zeros_(self.gate_net[-2].bias)
        nn.init.zeros_(self.refine_net[-1].weight)

    def _edge_response(self, feature_map):
        base = feature_map.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(base, self.sobel_x, padding=1)
        grad_y = F.conv2d(base, self.sobel_y, padding=1)
        edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        edge = edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return edge

    def forward(self, x_rgb, x_dsm, H, W):
        B, N, C = x_rgb.shape
        x_rgb = x_rgb.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_dsm = x_dsm.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        rgb_reduce = self.rgb_reduce(x_rgb)
        dsm_reduce = self.dsm_reduce(x_dsm)
        diff_reduce = torch.abs(rgb_reduce - dsm_reduce)
        prod_reduce = rgb_reduce * dsm_reduce

        edge_rgb = self._edge_response(x_rgb)
        edge_dsm = self._edge_response(x_dsm)
        edge_feat = self.edge_encoder(torch.cat([edge_rgb, edge_dsm, torch.abs(edge_rgb - edge_dsm)], dim=1))

        gate = self.gate_net(
            torch.cat([rgb_reduce, dsm_reduce, diff_reduce, prod_reduce, edge_feat], dim=1)
        )
        refine = self.refine_net(torch.cat([diff_reduce, prod_reduce, edge_feat], dim=1))
        edge_weight = torch.sigmoid(edge_rgb + edge_dsm)

        fused = gate * x_rgb + (1.0 - gate) * x_dsm + self.refine_scale * edge_weight * refine
        return fused.flatten(2).transpose(1, 2).contiguous()


def _normalize_range(value, default):
    if value is None:
        return default
    if isinstance(value, (tuple, list)) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
    else:
        low, high = default
    return (min(low, high), max(low, high))


def _sample_block_hw(height, width, block_scale, aspect_ratio, max_retry=10):
    min_scale, max_scale = _normalize_range(block_scale, (0.1, 0.3))
    min_aspect, max_aspect = _normalize_range(aspect_ratio, (0.5, 2.0))
    area = height * width

    for _ in range(max_retry):
        target_area = random.uniform(min_scale, max_scale) * area
        aspect = random.uniform(min_aspect, max_aspect)
        block_h = int(round(math.sqrt(target_area / aspect)))
        block_w = int(round(math.sqrt(target_area * aspect)))
        if 1 <= block_h <= height and 1 <= block_w <= width:
            return block_h, block_w

    fallback_h = max(1, min(height, int(round(height * math.sqrt(min_scale)))))
    fallback_w = max(1, min(width, int(round(width * math.sqrt(min_scale)))))
    return fallback_h, fallback_w


def _apply_local_mask(tensor, indices, num_blocks, block_scale, aspect_ratio):
    if indices.numel() == 0:
        return
    _, _, height, width = tensor.shape
    num_blocks = max(1, int(num_blocks))
    for idx in indices.tolist():
        for _ in range(num_blocks):
            block_h, block_w = _sample_block_hw(height, width, block_scale, aspect_ratio)
            top = 0 if height == block_h else random.randint(0, height - block_h)
            left = 0 if width == block_w else random.randint(0, width - block_w)
            tensor[idx, :, top:top + block_h, left:left + block_w] = 0


def apply_mcrc_mask(
    rgb,
    dsm,
    ratio=0.5,
    num_blocks=1,
    block_scale=(0.1, 0.3),
    aspect_ratio=(0.5, 2.0),
):
    batch_size = rgb.size(0)
    if batch_size == 0 or ratio <= 0:
        return rgb, dsm

    num_mask = int(batch_size * ratio)
    if num_mask == 0:
        return rgb, dsm

    if dsm.dim() == 3:
        dsm_in = dsm.unsqueeze(1)
    else:
        dsm_in = dsm

    perm = torch.randperm(batch_size, device=rgb.device)
    mask_idx = perm[:num_mask]
    split = num_mask // 2
    rgb_idx = mask_idx[:split]
    dsm_idx = mask_idx[split:]

    _apply_local_mask(rgb, rgb_idx, num_blocks, block_scale, aspect_ratio)
    _apply_local_mask(dsm_in, dsm_idx, num_blocks, block_scale, aspect_ratio)

    if dsm.dim() == 3:
        dsm = dsm_in.squeeze(1)
    else:
        dsm = dsm_in
    return rgb, dsm
