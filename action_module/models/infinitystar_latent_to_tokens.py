from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.vae96_to_tsformer_adapter import Vae96ToTSformerEmbedAdapter


class InfinityStarLatentToTokens(nn.Module):
    """
    End-to-end module: latent -> InfinityStar VAE decode (hook up_block_3) -> Adapter -> TSformer patch tokens.

    Input:
      z_ext: (B, 64, T_lat, 16, 16)  (the same format produced by our latent encoder script)

    Output:
      tokens: (B*T, 480, 384)
      T: number of decoded frames
      W_grid: 40
    """

    def __init__(self, vae: nn.Module, adapter: Optional[Vae96ToTSformerEmbedAdapter] = None):
        super().__init__()
        self.vae = vae
        self.adapter = adapter if adapter is not None else Vae96ToTSformerEmbedAdapter()

    @torch.no_grad()
    def forward(self, z_ext: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if z_ext.ndim != 5:
            raise ValueError(f"expected z_ext (B,64,T_lat,16,16), got {tuple(z_ext.shape)}")
        B = int(z_ext.shape[0])

        # We must know total decoded frames T to assemble the correct BT ordering.
        # Run teacher VAE decode once, collecting up3 features slice-by-slice.
        tokens_btnd = None
        meta: Dict[str, int] = {"T": 0, "N": 0, "D": 0, "W_grid": 0}
        start = 0

        def hook(_module, _inp, out):
            nonlocal tokens_btnd, start, meta
            hs = out[0] if isinstance(out, (tuple, list)) else out  # (B,96,t_slice,H,W)
            if not isinstance(hs, torch.Tensor) or hs.ndim != 5:
                raise RuntimeError("up_block_3 hook output is not a 5D Tensor")
            t_slice = int(hs.shape[2])

            # Adapter expects gradients in training, but this wrapper is inference-only.
            tok, _t2, wgrid = self.adapter(hs)  # (B*t_slice,480,384)
            _, N, D = tok.shape

            if tokens_btnd is None:
                # We cannot allocate full (B,T,...) yet because T is only known after full decode.
                # So we temporarily store per-slice tokens; allocate after decode and reorder.
                tokens_btnd = []
                meta["N"] = int(N)
                meta["D"] = int(D)
                meta["W_grid"] = int(wgrid)

            tok_btnd = tok.view(B, t_slice, N, D).contiguous()
            tokens_btnd.append((start, tok_btnd))
            start += t_slice

        handle = self.vae.decoder.up_blocks[-1].register_forward_hook(hook)
        try:
            _ = self.vae.decode(z_ext, return_dict=False)[0]
        finally:
            handle.remove()

        if tokens_btnd is None:
            raise RuntimeError("No up_block_3 features captured; check hook location")

        T = int(start)
        meta["T"] = T

        # Allocate and fill in correct time order, then flatten to (B*T,N,D) with (b,t) order.
        out = torch.empty(
            (B, T, meta["N"], meta["D"]),
            device=next(self.adapter.parameters()).device,
            dtype=next(self.adapter.parameters()).dtype,
        )
        for s, tok_btnd in tokens_btnd:
            out[:, s : s + tok_btnd.shape[1]] = tok_btnd
        out = out.reshape(B * T, meta["N"], meta["D"]).contiguous()
        return out, T, meta["W_grid"]

