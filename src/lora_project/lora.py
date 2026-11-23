"""Własna implementacja LoRA (Hu et al., 2021) - SZKIELET do uzupełnienia.

h = W x + (alpha / r) * B A x
    W: [d_out, d_in]  zamrożone
    A: [r, d_in]      trenowalne, mała losowa inicjalizacja
    B: [d_out, r]     trenowalne, inicjalizacja zerami  (=> dW = 0 na starcie)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Owija istniejący nn.Linear: zamrożone W + trenowalna poprawka niskorangowa BA."""

    def __init__(self, base_linear: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        # TODO: zapamiętaj base_linear i zamroź jego parametry (requires_grad=False)
        # TODO: in_dim / out_dim z base_linear
        # TODO: A = nn.Parameter([rank, in_dim]) - losowa inicjalizacja (np. Kaiming lub randn * 0.01)
        # TODO: B = nn.Parameter([out_dim, rank]) - zera
        # TODO: self.scale = alpha / rank
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, in_dim] -> [B, T, out_dim].

        TODO: original = base(x); low_rank = x @ A^T  -> [B, T, r];
              update = low_rank @ B^T -> [B, T, out_dim]; return original + scale * update
        """
        raise NotImplementedError

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """TODO: zwróć W + scale * (B @ A) [d_out, d_in] (do scalania po treningu)."""
        raise NotImplementedError


def apply_lora(
    model: nn.Module,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    rank: int = 4,
    alpha: float = 8.0,
    layer_indices: list[int] | None = None,
) -> nn.Module:
    """Podmienia wybrane nn.Linear w modelu na LoRALinear.

    TODO: iteruj po model.model.decoder.layers (OPT), dla warstw z layer_indices
          (None = wszystkie) podmień block.self_attn.<name> dla name w target_modules.
    TODO: zamroź wszystkie pozostałe parametry modelu (głowicę klasyfikacyjną zostaw trenowalną).
    """
    raise NotImplementedError


def merge_lora(model: nn.Module) -> nn.Module:
    """TODO: zamień każdy LoRALinear z powrotem na nn.Linear z W' = W + scale*BA."""
    raise NotImplementedError


def count_parameters(model: nn.Module) -> dict[str, int | float]:
    """TODO: zwróć {'trainable': ..., 'total': ..., 'trainable_pct': ...}."""
    raise NotImplementedError
