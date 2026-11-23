"""Testy poprawności LoRALinear. Zdefiniowane z góry (TDD) - będą czerwone do czasu implementacji."""
import torch
import torch.nn as nn

from lora_project import LoRALinear, count_parameters


def make(rank=4, alpha=8.0, d_in=16, d_out=12):
    torch.manual_seed(0)
    base = nn.Linear(d_in, d_out)
    return base, LoRALinear(base, rank=rank, alpha=alpha)


def test_identity_at_init():
    base, lora = make()
    x = torch.randn(2, 5, 16)
    assert torch.allclose(lora(x), base(x), atol=1e-6)  # B = 0 => dW = 0


def test_output_shape():
    _, lora = make()
    assert lora(torch.randn(2, 5, 16)).shape == (2, 5, 12)


def test_only_A_and_B_trainable():
    _, lora = make(rank=4)
    names = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert names == {"A", "B"}
    assert count_parameters(lora)["trainable"] == 4 * (16 + 12)  # r * (d_in + d_out)


def test_gradients_only_in_A_B():
    base, lora = make()
    with torch.no_grad():
        lora.B.normal_()  # B=0 => grad A = 0; losujemy B, by sprawdzić przepływ
    lora(torch.randn(3, 16)).sum().backward()
    assert base.weight.grad is None
    assert lora.A.grad is not None and lora.B.grad is not None


def test_merge_equals_forward():
    base, lora = make()
    with torch.no_grad():
        lora.B.normal_()
    x = torch.randn(3, 16)
    merged = nn.functional.linear(x, lora.merged_weight(), base.bias)
    assert torch.allclose(lora(x), merged, atol=1e-5)
