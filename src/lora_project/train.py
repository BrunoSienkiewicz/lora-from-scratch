"""Pętla treningowa - SZKIELET."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """TODO: random, numpy, torch (+ cuda) - reprodukowalność."""
    raise NotImplementedError


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device) -> float:
    """TODO: forward -> cross_entropy(logits, labels) -> backward -> step; zwróć średnią stratę.
    Optymalizator powinien dostać TYLKO parametry z requires_grad=True."""
    raise NotImplementedError


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device) -> dict[str, float]:
    """TODO: model.eval(), zbierz logity i etykiety, zwróć metrics.classification_metrics."""
    raise NotImplementedError


def run_experiment(config: dict) -> dict:
    """TODO: złóż całość wg configs/*.yaml (mode: head_only | lora | full) i zwróć wyniki
    (metryki + liczba parametrów + czas + VRAM) do tabeli ablacji."""
    raise NotImplementedError
