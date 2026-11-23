# LoRA od środka: efektywne dostrajanie małego modelu językowego

Projekt zespołowy (pary) z przedmiotu GSN 26Z, temat 14. Celem jest zrozumienie PEFT przez **samodzielną implementację LoRA**, a nie użycie gotowej biblioteki.

> Status: szkielet. Klasy w `src/lora_project/` mają sygnatury i TODO, bez implementacji. Testy w `tests/` są napisane z góry i na razie czerwone.

## Opis projektu

Pełny fine-tuning trenuje wszystkie parametry modelu (gradienty i stany optymalizatora kosztują pamięć). LoRA zamraża pretrenowaną macierz `W` i uczy tylko niskorangową poprawkę:

```
h = W x + (alpha / r) * B A x
W: [d_out, d_in] (zamrożone),  A: [r, d_in],  B: [d_out, r],  r << d
```

`B` startuje od zer, więc na początku model jest identyczny z pretrenowanym. Liczba trenowanych parametrów na warstwę spada z `d_in*d_out` do `r*(d_in+d_out)`.

Co robimy:
1. Bierzemy mały pretrenowany Transformer (OPT-125M).
2. Implementujemy własny `LoRALinear` (zamrożone `W`, trenowalne `A`, `B`).
3. Wpinamy go do `q_proj` i `v_proj` w blokach decodera.
4. Dostrajamy na małym zadaniu tekstowym (kilka do kilkunastu tysięcy krótkich tekstów).
5. Porównujemy z baseline'em head-only.

## Założenia (do potwierdzenia na konsultacjach)

- **Model:** `facebook/opt-125m` (decoder, causal), głowica klasyfikacyjna na reprezentacji tekstu (ostatni token nie-paddingowy, nie `[CLS]`).
- **Zadanie:** klasyfikacja tekstu, zbiór do wyboru (np. SST-2, AG News). Strata: cross-entropy.
- **Baseline:** head-only (zamrożony backbone, trenowana tylko głowica).
- **Rozszerzenie:** własna LoRA w `q_proj`/`v_proj`.
- **Ablacje:** ranga `r` (np. 1, 2, 4, 8, 16, 32), wybór warstw z LoRA (np. tylko q, q+v, wszystkie).
- **Metryki jakości:** accuracy i macro-F1 na zbiorze testowym; wybór modelu na walidacyjnym.
- **Metryki kosztu:** liczba trenowanych parametrów, czas treningu, szczytowe VRAM.
- **`peft`:** tylko do weryfikacji poprawności własnej implementacji, nie jako implementacja.
- **Reprodukowalność:** stałe seedy, konfiguracje w `configs/*.yaml`.
- Wynik negatywny jest akceptowalny, jeśli eksperyment jest poprawny i umiemy go przeanalizować.

## Otwarte pytania

- [ ] Jakie konkretnie zadanie i zbiór danych? Wybieramy sami czy są narzucone?
- [ ] Czy baseline head-only wystarczy, czy wymagany jest też pełny fine-tuning jako punkt odniesienia?
- [ ] Czy `peft` wolno używać do weryfikacji własnej implementacji?
- [ ] Które rangi i które warstwy są obowiązkowe w ablacji, a które opcjonalne?
- [ ] Jaki sprzęt: wystarczy CPU/Colab, czy oczekiwany jest pomiar VRAM na GPU?
- [ ] Jakie metryki są wymagane (accuracy, F1, inne)?
- [ ] Co dokładnie oddajemy na spotkaniu baseline'u (kod, notebook, raport)?
- [ ] Format i terminy obrony oraz sposób demonstracji.
- [ ] Podział pracy w parze (przy założeniu, że oboje rozumiemy całość).
- [ ] Jak ustawić `padding_side` i pooling dla decodera przy klasyfikacji?

## Struktura repozytorium

```
.
├── README.md
├── pyproject.toml                 # zależności (uv)
├── configs/baseline_head_only.yaml
├── notebooks/00_wstepne_zrozumienie.ipynb   # materiał na konsultacje
├── src/lora_project/
│   ├── lora.py       # LoRALinear, apply_lora, merge_lora, count_parameters
│   ├── model.py      # OPTClassifier, freeze_backbone
│   ├── data.py       # load_task, make_loaders
│   ├── metrics.py    # classification_metrics, CostTracker
│   └── train.py      # set_seed, train_one_epoch, evaluate, run_experiment
└── tests/test_lora.py             # testy poprawności LoRALinear
```

## Uruchomienie

```bash
uv sync --extra dev
uv run pytest            # na starcie czerwone (NotImplementedError)
uv run jupyter lab notebooks/00_wstepne_zrozumienie.ipynb
```

## Plan prac

1. Konsultacje wstępne: ustalenie zadania, danych i wymagań.
2. Baseline: OPT-125M z głowicą head-only i ewaluacja.
3. `LoRALinear` plus testy, wpięcie do modelu.
4. Eksperymenty: rangi, warstwy, koszt.
5. Analiza wyników i błędów, porównanie baseline vs LoRA.

## Literatura

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021), arXiv:2106.09685
- Zhang et al., *OPT: Open Pre-trained Transformer Language Models* (2022), arXiv:2205.01068
