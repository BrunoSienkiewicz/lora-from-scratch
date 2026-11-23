"""Dane i tokenizacja - SZKIELET. Zadanie i zbiór do ustalenia (zob. README)."""
from __future__ import annotations

from torch.utils.data import DataLoader


def load_task(task: str, max_len: int = 128, model_name: str = "facebook/opt-125m", seed: int = 42):
    """TODO: wczytaj zbiór (datasets), podziel na train/val/test (kilka-kilkanaście tys. przykładów),
    zatokenizuj (padding, truncation, attention_mask). Zwróć (train_ds, val_ds, test_ds)."""
    raise NotImplementedError


def make_loaders(train_ds, val_ds, test_ds, batch_size: int = 32) -> tuple[DataLoader, DataLoader, DataLoader]:
    """TODO: DataLoadery (shuffle tylko dla train)."""
    raise NotImplementedError
