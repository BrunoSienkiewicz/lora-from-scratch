"""Budowa modelu: OPT-125M + głowica klasyfikacyjna - SZKIELET."""
from __future__ import annotations

import torch.nn as nn


class OPTClassifier(nn.Module):
    """Pretrenowany OPT-125M (backbone) + liniowa głowica na reprezentacji tekstu.

    Wejście:  input_ids [B, T], attention_mask [B, T]
    Wyjście:  logits [B, num_labels]
    """

    def __init__(self, model_name: str = "facebook/opt-125m", num_labels: int = 2):
        super().__init__()
        # TODO: załaduj AutoModel (backbone) z transformers
        # TODO: nn.Linear(hidden_size, num_labels)
        raise NotImplementedError

    def forward(self, input_ids, attention_mask):
        # TODO: hidden = backbone(...).last_hidden_state  [B, T, d]
        # TODO: pooling - OPT jest decoderem (causal), więc użyj ostatniego NIE-paddingowego tokenu
        #       (uwaga na padding_side!), nie tokenu [CLS]
        # TODO: return self.head(pooled)
        raise NotImplementedError


def freeze_backbone(model: OPTClassifier) -> None:
    """TODO: baseline head-only - requires_grad=False dla backbone, trenowalna tylko głowica."""
    raise NotImplementedError
