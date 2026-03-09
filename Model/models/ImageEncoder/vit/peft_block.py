from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .peft_modules import Adapter as PeftAdapter
from .peft_modules import CPIABlock


class PeftFusionBlock(nn.Module):
    """Transformer block with PEFT prompt injection for dual-modality features."""

    def __init__(
        self,
        args,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

        peft_ratio = getattr(args, "peft_ratio", 0.25)
        adapter_ratio = getattr(args, "peft_adapter_ratio", 0.0625)
        self.peft_cpia = CPIABlock(dim, ratio=peft_ratio)
        self.peft_adapter_x = PeftAdapter(dim, mlp_ratio=adapter_ratio, prompt_add=True)
        self.peft_adapter_y = PeftAdapter(dim, mlp_ratio=adapter_ratio, prompt_add=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        shortcutx = x
        shortcuty = y
        x = self.norm1(x)
        y = self.norm1(y)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hwx = window_partition(x, self.window_size)
            y, pad_hwy = window_partition(y, self.window_size)

        x = self.attn(x)
        y = self.attn(y)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hwx, (H, W))
            y = window_unpartition(y, self.window_size, pad_hwy, (H, W))

        x = shortcutx + x
        y = shortcuty + y

        x = x + self.mlp(self.norm2(x))
        y = y + self.mlp(self.norm2(y))

        H, W = x.shape[1], x.shape[2]
        cpia_x, cpia_y = self.peft_cpia(x, y, H, W)
        x = self.peft_adapter_x(x, prompt=cpia_x)
        y = self.peft_adapter_y(y, prompt=cpia_y)

        return x, y


class PeftPromptFusion(nn.Module):
    """PEFT prompt fusion without replacing the backbone attention/MLP."""

    def __init__(self, args, dim: int, input_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        peft_ratio = getattr(args, "peft_ratio", 0.25)
        adapter_ratio = getattr(args, "peft_adapter_ratio", 0.0625)
        self.peft_cpia = CPIABlock(dim, ratio=peft_ratio)
        self.peft_adapter_x = PeftAdapter(dim, mlp_ratio=adapter_ratio, prompt_add=True)
        self.peft_adapter_y = PeftAdapter(dim, mlp_ratio=adapter_ratio, prompt_add=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor, H: int, W: int):
        # Accept either (B, N, C) or (B, H, W, C)
        if x.dim() == 3:
            B, N, C = x.shape
            x_hw = x.reshape(B, H, W, C)
            y_hw = y.reshape(B, H, W, C)
            flatten = True
        elif x.dim() == 4:
            B, H, W, C = x.shape
            x_hw = x
            y_hw = y
            flatten = False
        else:
            raise ValueError("Unsupported tensor shape for PEFT prompt fusion.")

        cpia_x, cpia_y = self.peft_cpia(x_hw, y_hw, H, W)
        x_hw = self.peft_adapter_x(x_hw, prompt=cpia_x)
        y_hw = self.peft_adapter_y(y_hw, prompt=cpia_y)

        if flatten:
            return x_hw.reshape(B, H * W, C), y_hw.reshape(B, H * W, C)
        return x_hw, y_hw


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            self.rel_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_h, self.rel_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn
