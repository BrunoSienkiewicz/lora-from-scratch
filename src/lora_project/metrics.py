"""Metryki jakości i kosztu - SZKIELET."""
from __future__ import annotations

import torch


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """TODO: accuracy, macro-F1 (do ustalenia z prowadzącym)."""
    raise NotImplementedError


class CostTracker:
    """TODO: czas treningu (s) i szczytowe VRAM (torch.cuda.max_memory_allocated; na CPU/MPS - brak/inna metoda)."""

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, *exc):
        raise NotImplementedError

    def result(self) -> dict[str, float]:
        raise NotImplementedError
