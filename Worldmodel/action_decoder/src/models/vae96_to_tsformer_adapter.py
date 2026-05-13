from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_to_192x640(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=(192, 640), mode="bilinear", align_corners=False)


class Vae96ToTSformerEmbedAdapter(nn.Module):
    """
    Map InfinityStar decoder `up_block_3` feature (B,96,T,256,256) to TSformer patch tokens.

    Output tokens match TSformer `PatchEmbed` output:
      - patch_tokens: (B*T, N=12*40=480, D=384)
      - returns (patch_tokens, T, W_grid=40)
    """

    def __init__(self, embed_dim: int = 384, patch_size: int = 16, use_skip: bool = False):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.use_skip = bool(use_skip)

        self.conv_a = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
        )

        self.patch = nn.Conv2d(128, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.skip = nn.Conv2d(96, self.embed_dim, kernel_size=1, padding=0, bias=False) if self.use_skip else None
        self.out_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, f96_up3: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if f96_up3.ndim != 5:
            raise ValueError(f"expected f96_up3 shape (B,96,T,H,W), got {tuple(f96_up3.shape)}")
        B, C, T, H, W = f96_up3.shape
        if int(C) != 96:
            raise ValueError(f"expected channel=96, got C={C}")

        x = f96_up3.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (BT,96,H,W)
        x = _resize_to_192x640(x)  # (BT,96,192,640)

        h = self.conv_a(x)  # (BT,128,192,640)
        h = self.patch(h)  # (BT,384,12,40)

        if self.skip is not None:
            s = self.skip(x)  # (BT,384,192,640)
            s = F.avg_pool2d(s, kernel_size=self.patch_size, stride=self.patch_size)  # (BT,384,12,40)
            xg = h + 0.1 * s
        else:
            xg = h

        tokens = xg.flatten(2).transpose(1, 2).contiguous()  # (BT,480,384)
        tokens = self.out_norm(tokens)
        grid_w = int(xg.shape[-1])
        return tokens, int(T), grid_w

