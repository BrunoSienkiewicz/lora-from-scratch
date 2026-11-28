# LoRA od środka: kategoryzacja filmów YouTube małym modelem językowym

Projekt zespołowy (Janek & Bruno) z przedmiotu **GSN 26Z**, temat 14. Samodzielnie implementujemy **LoRA** (Low-Rank Adaptation) i sprawdzamy, ile jakości da się zachować, ucząc ułamek parametrów modelu **OPT-125M** w zadaniu klasyfikacji filmów YouTube po tytule i opisie.

Cel zajęć to **zrozumienie PEFT przez własną implementację**, nie użycie biblioteki. Ocena zależy głównie od tego, czy umiemy wyjaśnić ustnie, co robi nasz kod (patrz [sekcja 11](#11-obrona-i-ocena)).

## Spis treści

1. [Projekt w skrócie](#1-projekt-w-skrócie)
2. [LoRA w pięć minut](#2-lora-w-pięć-minut)
3. [Dane](#3-dane)
4. [Model i baseline](#4-model-i-baseline)
5. [Plan eksperymentów](#5-plan-eksperymentów)
6. [Metryki i hiperparametry](#6-metryki-i-hiperparametry)
7. [Stos technologiczny](#7-stos-technologiczny)
8. [Struktura repozytorium i kontrakty](#8-struktura-repozytorium-i-kontrakty)
9. [Harmonogram i podział zadań](#9-harmonogram-i-podział-zadań)
10. [Zasady pracy](#10-zasady-pracy)
11. [Obrona i ocena](#11-obrona-i-ocena)
12. [Uruchomienie](#12-uruchomienie)
13. [Decyzje, ryzyka i sprawy do potwierdzenia](#13-decyzje-ryzyka-i-sprawy-do-potwierdzenia)
14. [Literatura](#14-literatura)

---

## 1. Projekt w skrócie

| | |
|---|---|
| **Zadanie** | Klasyfikacja filmu YouTube do jednej z 6 kategorii na podstawie tytułu i początku opisu |
| **Kategorie** | Music, Gaming, Entertainment, Science & Technology, Sports, News & Politics |
| **Dane** | Podzbiór *Trending YouTube Video Statistics*: 12 000 rekordów (8 000 train / 2 000 val / 2 000 test), po ok. 2 000 na kategorię |
| **Model** | `facebook/opt-125m` (decoder, causal) + liniowa głowica klasyfikacyjna |
| **Baseline** | *head-only*: backbone zamrożony, uczymy tylko głowicę |
| **Rozszerzenie** | Własny `LoRALinear` w `q_proj`/`v_proj` (oraz inne warstwy w ablacji) |
| **Pytania badawcze** | Ile jakości daje LoRA względem head-only? Jak wpływa ranga `r`? Które warstwy opłaca się adaptować? Ile to kosztuje (parametry, czas, VRAM)? |
| **Sprzęt** | 1 GPU (Colab/Kaggle T4), mixed precision `16-mixed` |
| **Terminy** | Oddanie baseline'u **18.12** (EOD), obrona P3 **21.12**. Oddanie rozszerzenia **22.01** (EOD), obrona P4 **25.01** |

Wynik negatywny jest akceptowalny (LoRA nie musi pobić baseline'u), jeśli eksperyment jest poprawny i potrafimy go przeanalizować.

---

## 2. LoRA w pięć minut

Pełny fine-tuning uczy każdą wagę modelu, więc dla każdej z nich trzeba trzymać gradient i dwa stany optymalizatora AdamW. LoRA **zamraża** pretrenowaną macierz `W` i uczy tylko niskorangową poprawkę:

```
h = W x + (alpha / r) · B A x

W : [d_out, d_in]   zamrożona (pretrenowana)
A : [r,     d_in]   trenowalna, init  A ~ N(0, 0.01²)
B : [d_out, r]      trenowalna, init  B = 0
r << d              ranga poprawki
```

Dlaczego to ma sens:

- **Start identyczny z modelem pretrenowanym.** `B = 0`, więc `ΔW = (alpha/r)·BA = 0` i na początku `h = W x`. Trening zaczyna się od wiedzy modelu, nie od szumu.
- **Mało parametrów.** Zamiast `d_in · d_out` uczymy `r · (d_in + d_out)`.
- **Zero narzutu po treningu.** Poprawkę można wcielić do wag: `W' = W + (alpha/r)·BA`. Model ma wtedy tę samą architekturę i te same FLOPs co oryginał.
- **Skalowanie `alpha / r`.** Dzięki niemu zmiana rangi nie zmienia drastycznie wielkości poprawki. W ablacji rangi trzymamy `alpha / r = 2`, czyli `alpha = 2r`.

### Kształty tensorów w `LoRALinear.forward`

```
x          [B, T, d_in]
base(x)    [B, T, d_out]        # W x + bias, zamrożone
x @ A^T    [B, T, r]            # projekcja w dół
… @ B^T    [B, T, d_out]        # projekcja w górę
wynik      base(x) + (alpha/r) · (x @ A^T @ B^T)
```

### Rachunek parametrów dla OPT-125M

OPT-125M ma 12 bloków decodera, `hidden = 768`, FFN `768 → 3072 → 768`, łącznie ok. **125 mln** parametrów. Przy LoRA w `q_proj` i `v_proj` (każda 768 × 768) w 12 blokach:

| Ranga `r` | Parametry LoRA (`12 · 2 · r · 1536`) | % całego modelu |
|---:|---:|---:|
| 2 | 73 728 | ok. 0,06 % |
| 4 | 147 456 | ok. 0,12 % |
| 8 | 294 912 | ok. 0,24 % |
| 16 | 589 824 | ok. 0,47 % |

Do tego dochodzi głowica klasyfikacyjna (`768 · 6 + 6 = 4 614` parametrów), która jest trenowana zawsze. Te liczby służą jako **kontrola poprawności**: `count_parameters` / `get_param_stats` muszą je odtworzyć.

---

## 3. Dane

- **Źródło:** *Trending YouTube Video Statistics* (Kaggle). Pola, których używamy: `title`, `description`, `category_id` (mapowanie na nazwę z plików `*_category_id.json`), `video_id`.
- **Wejście modelu:** jeden ciąg tekstowy `[TITLE] <tytuł> [DESC] <początek opisu>`. Znaczniki to zwykły tekst, nie dodajemy ich do słownika tokenizera. Tytuł jest przed opisem, więc przy ucięciu do `max_length = 128` tokenów traci się koniec opisu, nie tytuł.
- **Czyszczenie:** usuwamy adresy URL i surowe tagi HTML, normalizujemy białe znaki.
- **Etykieta:** jedna z 6 kategorii, mapowanie `nazwa → id` zapisane raz i współdzielone przez wszystkie eksperymenty.
- **Zrównoważenie i split:** po 2 000 przykładów na kategorię, podział 8 000 / 2 000 / 2 000 (stratyfikowany).
- **Split deterministyczny:** `seed = 42`, zapisany w `splits/splits.json` (lista `video_id` na split) i wspólny dla baseline'u i wszystkich runów LoRA. Dzięki temu porównania są uczciwe.
- **Brak wycieku:** zbiór *Trending* zawiera ten sam film wielokrotnie (kolejne dni w trendach, kilka krajów). Dlatego **deduplikujemy po `video_id` przed splitem**, a split robimy po `video_id`. Bez tego ten sam film trafiłby do train i test, a metryki byłyby zawyżone.
- **Języki:** OPT jest modelem angielskim. Używamy plików krajów anglojęzycznych (US, GB, CA). Kategorie, w których zabraknie 2 000 unikalnych filmów, trzeba rozwiązać przed startem (patrz [sekcja 13](#13-decyzje-ryzyka-i-sprawy-do-potwierdzenia)).

---

## 4. Model i baseline

### `OPTClassifier`

```
input_ids [B, T] ─┐
attention_mask    ├─► OPT-125M (backbone) ─► last_hidden_state [B, T, 768]
[B, T]           ─┘                                 │
                          pooling: wektor ostatniego NIE-paddingowego tokenu
                                                    ▼
                                             pooled [B, 768]
                                                    │
                                     nn.Linear(768, 6) (głowica)
                                                    ▼
                                           logits [B, 6]
```

- **Dlaczego ostatni token, a nie `[CLS]`?** OPT to decoder z maską przyczynową: token widzi tylko tokeny sprzed siebie, więc dopiero **ostatni** token „widział" cały tekst. Tokenu `[CLS]` OPT nie ma.
- **Padding.** Tokenizer OPT domyślnie dopełnia z prawej. Indeks ostatniego realnego tokenu liczymy jako `attention_mask.sum(dim=1) - 1` i pobieramy po nim wektor z `[B, T, 768]`. Przy paddingu z prawej tokeny pad nie wpływają na wcześniejsze (maska przyczynowa), więc to jest poprawne. Nie używamy `hidden[:, -1]`, bo to byłby często token pad.
- **Strata:** `CrossEntropyLoss` na logitach (wewnątrz softmax), etykiety `[B]` jako indeksy klas. Klasyfikacja wieloklasowa z jedną poprawną klasą.

### Tryby uczenia

| Tryb | Trenowane parametry | Rola |
|---|---|---|
| `head_only` | tylko głowica (4 614) | **Baseline.** Dolny punkt odniesienia |
| `lora` | macierze `A`, `B` w wybranych warstwach + głowica | **Rozszerzenie** |
| `full` | cały model (ok. 125 mln) | Górna granica kosztu (jeden kontrolny run) |

W trybie `lora` backbone jest zamrożony, a **głowica pozostaje trenowalna**: bez niej nie byłoby czego uczyć na wyjściu.

### `LoRALinear`: wymagania

- zamrożone `W` (i bias), trenowalne `A` i `B`, init `A ~ N(0, 0.01²)`, `B = 0`,
- skalowanie `alpha / r`, opcjonalny dropout na wejściu poprawki (domyślnie 0,05),
- `merged_weight()` zwraca `W + (alpha/r)·BA`, a `merge_lora(model)` podmienia każdy `LoRALinear` z powrotem na `nn.Linear` z wagą scaloną,
- brak zależności od `peft` w kodzie źródłowym.

Uwaga: oryginalny artykuł inicjalizuje `A` metodą Kaiming. My używamy `N(0, 0.01²)`, zgodnie ze szkicem z materiałów kursu. Wynik jest równoważny pod względem własności `ΔW = 0` na starcie.

---

## 5. Plan eksperymentów

Wszystkie runy używają tego samego splitu, tego samego tokenizera, tej samej metryki i tego samego typu GPU.

| # | Seria | Konfiguracja | Runy | Grupa W&B |
|---|---|---|---|---|
| E0 | Baseline | head-only | 3 seedy | `baseline-head-only` |
| E1 | Ablacja rangi | `q_proj + v_proj`, `r ∈ {2, 4, 8, 16}`, `alpha = 2r` | 4 | `ablation-rank` |
| E2 | Ablacja warstw | dla najlepszego `r` z E1: `q`; `q+v`; `q+k+v+out_proj` (cały self-attention); opcjonalnie attention + FFN (`fc1`, `fc2`) | 3–4 | `ablation-targets` |
| E3 | Górna granica kosztu | pełny fine-tuning (100 % wag OPT) | 1 | `ablation-targets` |

Zasady:

- Wartość `q+v` w E2 to ten sam run co najlepszy z E1, więc nie liczymy go drugi raz.
- Różnice między rangami bywają małe. Dlatego po E1 najlepszą konfigurację i jej najbliższego konkurenta powtarzamy na 3 seedach, żeby odróżnić efekt od szumu.
- Strojenie LR i `batch_size` robimy **na zbiorze walidacyjnym** dla baseline'u (J1.4) i potem stosujemy tę samą siatkę dla LoRA. Zbioru testowego używamy tylko do raportowania końcowego.
- Po serii ablacji robimy analizę błędów (baseline vs najlepsza LoRA) oraz analizę kosztów (sekcja 6).

---

## 6. Metryki i hiperparametry

### Jakość

- **Accuracy** i **macro-F1** na teście. Macro-F1 uśrednia F1 po klasach bez ważenia, więc nie pozwala ukryć słabej klasy (np. Gaming). Zbiór jest zrównoważony, więc obie miary powinny być zbliżone. Duża różnica byłaby sygnałem problemu.
- Wybór najlepszego checkpointu: `val/f1`.
- Macierz pomyłek dla kategorii YouTube.

### Koszt

- liczba trenowanych parametrów i ich % (`trainable_params`, `trainable_pct`),
- czas epoki i czas całego treningu, przepustowość (`samples/sec`),
- szczytowe VRAM (`peak_vram_mb`, `torch.cuda.max_memory_allocated`),
- FLOPs/MACs (`fvcore`) dla baseline'u i LoRA przed i po scaleniu wag. Oczekujemy, że po `merge_lora` koszt równa się modelowi bazowemu.

### Domyślne hiperparametry (punkt startowy)

| Parametr | Wartość | Uwagi |
|---|---|---|
| Optymalizator | AdamW, `weight_decay = 0.0` | Optymalizator dostaje tylko parametry z `requires_grad=True` |
| Harmonogram LR | liniowy spadek, warmup 5 % | |
| Learning rate | siatka `{1e-4, 3e-4, 1e-3}` | head-only i LoRA: wyższy LR; `full`: ok. `2e-5` |
| Epoki | 3 | |
| Batch size | 32 | Dobrany do T4, zmieniany tylko wspólnie dla wszystkich runów |
| `max_length` | 128 | |
| Precyzja | `16-mixed` | T4 nie obsługuje bf16 |
| LoRA dropout | 0,05 | |
| Seed | 42 (+ 2 dodatkowe dla powtórzeń) | |

Wartości są punktem startu. Ostateczne ustala J1.4 i zapisuje w `configs/`.

### Dlaczego redukcja VRAM jest mniejsza niż redukcja liczby parametrów?

To pytanie z analizy J2.5, a jednocześnie typowe pytanie na obronie. LoRA oszczędza gradienty i stany AdamW tylko dla zamrożonych wag (te stany to ok. 2× rozmiar trenowanych parametrów). Natomiast:

- wagi modelu dalej leżą w pamięci w całości,
- **aktywacje** nadal trzeba przechowywać dla całej sieci, bo gradient musi przepłynąć wstecz przez wszystkie zamrożone warstwy do warstw z LoRA.

Dlatego VRAM spada wyraźnie mniej niż liczba trenowanych parametrów (0,24 % vs 100 %). J2.5 rozkłada pamięć na: wagi / stany optymalizatora / aktywacje.

---

## 7. Stos technologiczny

| Narzędzie | Właściciel | Rola |
|---|---|---|
| **PyTorch** | Janek | `OPTClassifier`, `LoRALinear`, wstrzykiwanie adapterów, scalanie wag |
| **PyTorch Lightning** | Bruno | `YouTubeDataModule` (dane) i `LitVideoClassifier` (pętla treningowa), `16-mixed`, callbacki profilujące |
| **Hydra** | Bruno (szkielet), Janek (modele) | Kompozycja konfiguracji YAML, serie ablacji jednym poleceniem (`-m` / multirun) |
| **Weights & Biases** | Bruno (dashboard), obaj (runy) | Metryki, VRAM, czas epoki, tabele błędów (`wandb.Table`) |
| **`torchtest2` + `jaxtyping`** | Janek | Testy niezmienników treningu (`W` zamrożone, `A`/`B` się zmieniają, brak `NaN`/`Inf`) i adnotacje kształtów `[B, T, d]` |
| **`peft`** | Janek | **Tylko w testach** jako wyrocznia: zgodność numeryczna z własnym `LoRALinear` przy identycznych wagach |
| **`fvcore`** | Janek | FLOPs/MACs: baseline vs LoRA, przed i po scaleniu |
| **`cleanlab`** | Bruno | Wykrywanie błędnych etykiet i niejednoznacznych opisów w analizie błędów |
| **`ruff`** | wspólnie | Formatowanie i lint |

---

## 8. Struktura repozytorium i kontrakty

### Układ katalogów i właściciele

Podział na katalogi z jednym właścicielem ma zapobiegać konfliktom w Gicie.

```text
lora-from-scratch/
├── configs/                  # Hydra
│   ├── config.yaml           # [wspólne] główny plik składający (defaults)
│   ├── data/                 # [Bruno]   youtube.yaml (ścieżki, split, tokenizer, max_len, batch_size)
│   ├── model/                # [Janek]   opt_head_only.yaml, opt_lora.yaml
│   ├── lora/                 # [Janek]   r2.yaml, r4.yaml, r8.yaml, r16.yaml, targets_*.yaml
│   ├── trainer/              # [Bruno]   colab_t4.yaml, debug_cpu.yaml
│   ├── logger/               # [Bruno]   wandb.yaml
│   └── experiment/           # [wspólne] presety (e0_baseline, e1_lora_r8, ...)
├── splits/splits.json        # [Bruno]   deterministyczny split (seed=42), w repo
├── src/
│   ├── data/                 # [Bruno]   datamodule.py (YouTubeDataModule), preprocessing.py
│   ├── models/               # [Janek]   backbone.py, lora_linear.py, lora_injector.py
│   ├── training/             # [Bruno]   lit_module.py (LitVideoClassifier), metrics.py
│   ├── callbacks/            # [Bruno]   profiling.py (VRAM, czas), error_logger.py (W&B Table)
│   └── analysis/             # [Bruno]   error_analysis.py (cleanlab), export_wandb_tables.py
├── tests/                    # [Janek]   test_baseline_torchtest2.py, test_lora_torchtest2.py,
│                             #           test_lora.py, test_peft_parity.py
├── scripts/                  # [Bruno]   train.py, eval.py (entrypointy Hydra)
├── notebooks/                # [wspólne] 00_wstepne_zrozumienie.ipynb, run_colab.ipynb (reprodukcja na 1 GPU)
├── pyproject.toml            # [wspólne] zależności + konfiguracja ruff
└── README.md                 # [wspólne] ten dokument
```

### Obecny stan repozytorium

Repozytorium zawiera na razie **szkielet bez implementacji**: klasy mają sygnatury i `TODO`, a testy w `tests/test_lora.py` są napisane z góry i czerwone (`NotImplementedError`). Szkielet leży w płaskim pakiecie `src/lora_project/` i **w zadaniu E0.1 migrujemy go do układu powyżej**:

| Szkielet (`src/lora_project/`) | Docelowo |
|---|---|
| `lora.py`: `LoRALinear` | `src/models/lora_linear.py` |
| `lora.py`: `apply_lora`, `merge_lora`, `count_parameters` | `src/models/lora_injector.py` (jako `inject_lora`, `merge_lora`, `get_param_stats`) |
| `model.py`: `OPTClassifier`, `freeze_backbone` | `src/models/backbone.py` |
| `data.py` | `src/data/datamodule.py`, `preprocessing.py` |
| `metrics.py`: `classification_metrics`, `CostTracker` | `src/training/metrics.py`, `src/callbacks/profiling.py` |
| `train.py` | `src/training/lit_module.py`, `scripts/train.py` |
| `configs/baseline_head_only.yaml` | `configs/` w układzie Hydra |
| `tests/test_lora.py` | zostaje (importy do zmiany po migracji) |

Nazewnictwo ustalone na stałe: `LoRALinear.merged_weight()` zwraca tensor `W + (alpha/r)·BA`, `merge_lora(model)` podmienia moduły, a `get_param_stats` zwraca `total_params`, `trainable_params`, `trainable_pct`.

### Kontrakty interfejsów (zamrożone przed startem równoległej pracy)

1. **Dane → trening.** `YouTubeDataModule` zwraca batch jako `dict[str, torch.Tensor]`: `input_ids` `[B, T]` (`long`), `attention_mask` `[B, T]` (`long`), `labels` `[B]` (`long`), gdzie `T = 128`.
2. **Model.** `build_model(cfg)` przyjmuje konfigurację Hydra i zwraca `nn.Module` z `forward(input_ids, attention_mask) -> logits [B, num_classes]`. Funkcja `get_param_stats(model) -> dict` zwraca `total_params`, `trainable_params`, `trainable_pct`.
3. **Split.** `splits/splits.json` (`seed = 42`) jest współdzielony przez baseline i wszystkie runy LoRA.
4. **Konwencja W&B.** Projekt `youtube-lora-opt125m`, grupy `baseline-head-only`, `ablation-rank`, `ablation-targets`. Obowiązkowe tagi: `stage`, `rank`, `targets`, `seed`.

---

## 9. Harmonogram i podział zadań

Zależności oznaczają, co musi być gotowe przed startem zadania.

### Etap 0: inicjalizacja (27.11 – 29.11)

- **E0.1 [wspólnie, do 29.11]:** repozytorium, `pyproject.toml` (nowe zależności: `lightning`, `hydra-core`, `wandb`, `torchmetrics`, `torchtest2`, `jaxtyping`, `fvcore`, `cleanlab`, `ruff`), migracja szkieletu do docelowej struktury, projekt W&B, kontrakty z sekcji 8 w tym README.

### Etap 1: baseline (30.11 – 18.12, obrona P3: 21.12)

| ID | Kto | Zadanie | Zależność | Termin | Definition of Done |
|---|---|---|---|---|---|
| B1.1 | Bruno | `YouTubeDataModule`: pobranie danych, czyszczenie URL/HTML, deduplikacja po `video_id`, split 8k/2k/2k, tokenizacja OPT | E0.1 | 03.12 | `setup()` zwraca powtarzalne batche `[B, 128]` bez wycieku między splitami |
| J1.1 | Janek | `OPTClassifier` (z `jaxtyping`): ładowanie OPT-125M, pooling ostatniego tokenu, głowica, `freeze_backbone()` | E0.1 | 03.12 | Logity `[B, num_classes]`, `get_param_stats()` potwierdza zamrożenie 100 % wag OPT |
| B1.2 | Bruno | Szkielet Hydra (`config.yaml`, `data/`, `trainer/`, `logger/`) i `scripts/train.py` | B1.1 | 06.12 | `python scripts/train.py trainer.fast_dev_run=true` działa na mocku modelu |
| J1.2 | Janek | Testy `torchtest2` dla baseline'u: `assert_model_ok` ze sprawdzeniami `frozen_params` (OPT), `params_change` (głowica), `output_finite`, `output_shape`; bazowy pomiar FLOPs (`fvcore`) | J1.1 | 06.12 | Asercje przechodzą w `pytest`, zapisany raport FLOPs |
| B1.3 | Bruno | `LitVideoClassifier`: `CrossEntropyLoss`, `Accuracy` i `Macro-F1` (`torchmetrics`), `WandbLogger`, `ModelCheckpoint` | B1.2 | 09.12 | W&B loguje `train/loss`, `val/loss`, `val/acc`, `val/f1`, `test/f1` |
| J1.3 | Janek | `configs/model/opt_head_only.yaml` + test integracyjny `OPTClassifier` z `LitVideoClassifier` | J1.1, B1.3 | 10.12 | Przebieg dane → model → loss → W&B jednym poleceniem |
| B1.4 | Bruno | Callback profilujący: `peak_vram_mb`, czas epoki, `samples/sec`, statystyki parametrów w `wandb.summary` | B1.3 | 12.12 | Każdy run ma komplet kolumn kosztowych |
| J1.4 | Janek | Strojenie LR i `batch_size` dla baseline'u na T4 (`16-mixed`), 3 seedy kontrolne | J1.3, B1.4 | 15.12 | Stabilny wynik odniesienia na walidacji i teście |
| B1.5 | Bruno | `notebooks/run_colab.ipynb` (reprodukcja 1-click) i `scripts/eval.py` z macierzą pomyłek w W&B | J1.4 | 16.12 | Świeża sesja Colab T4 odtwarza trening i ewaluację bez ręcznej ingerencji |
| E1.1 | wspólnie | Freeze kodu P3, wzajemne code review, próba generalna z zamianą ról (17.12), oddanie baseline'u | B1.5 | **18.12 EOD** | Tag `v1.0-baseline` na `main`, projekt przesłany przed terminem |

### Etap 2: LoRA, ablacje, analiza (22.12 – 22.01, obrona P4: 25.01)

| ID | Kto | Zadanie | Zależność | Termin | Definition of Done |
|---|---|---|---|---|---|
| J2.1 | Janek | `LoRALinear` (`jaxtyping`): zamrożone `W`, `A ~ N(0, 0.01²)`, `B = 0`, skalowanie `alpha/r`, dropout, `merged_weight()` | E1.1 | 02.01 | Czysty PyTorch, zero zależności od `peft`, `tests/test_lora.py` zielone |
| B2.1 | Bruno | `LitVideoClassifier` przekazuje do AdamW tylko parametry `requires_grad=True`, obsługa multirun | E1.1 | 02.01 | Optymalizator aktualizuje wyłącznie `A`, `B` i głowicę |
| J2.2 | Janek | `inject_lora()` podmieniająca wskazane warstwy dekodera + `configs/lora/*.yaml` | J2.1 | 05.01 | `lora.rank=...` i `lora.targets=...` z CLI zmieniają architekturę i raportują liczbę parametrów |
| B2.2 | Bruno | Callback `error_logger.py`: błędne predykcje z testu (tytuł, opis, prawdziwa klasa, predykcja, pewność) do `wandb.Table` | B2.1 | 05.01 | Po teście w W&B jest przeszukiwalna tabela błędów |
| J2.3 | Janek | Testy LoRA: `test_lora_torchtest2.py` (`frozen_params` dla `W`, `params_change` dla `A`/`B`, `output_finite`, `assert_never_nan`/`assert_never_inf`, plus własna asercja: wyjście identyczne w kroku 0), `test_peft_parity.py`, FLOPs przed/po scaleniu | J2.2 | 07.01 | `pytest` zielony, zerowy narzut FLOPs po scaleniu |
| B2.3 | Bruno | `export_wandb_tables.py`: wyniki z W&B API, wykresy `F1 vs rank`, `F1 vs trainable_params`, `peak_vram_mb vs wariant` | B2.2 | 09.01 | Skrypt buduje tabele i wykresy do raportu |
| J2.4 | Janek | **Seria 1** (E1): rangi `r ∈ {2, 4, 8, 16}` na T4 przez `-m` | J2.3, B2.1 | 12.01 | 4 runy w `ablation-rank` z pełnymi metrykami i profilem VRAM |
| B2.4 | Bruno | **Seria 2** (E2 + E3): warstwy `q`, `q+v`, `q+k+v+o`, `+FFN` i kontrolnie `Full FT` dla optymalnego `r` | J2.4 | 15.01 | 4–5 runów w `ablation-targets` na tym samym typie GPU |
| B2.5 | Bruno | Analiza błędów (`cleanlab` + `wandb.Table`): baseline vs najlepsza LoRA, szum etykiet (np. Gaming vs Entertainment), wpływ ucięcia opisów przy `max_len = 128` | B2.3, B2.4 | 18.01 | 15 sklasyfikowanych przykładów błędów, wnioski o granicach modelu czysto tekstowego |
| J2.5 | Janek | Analiza mechanizmu i kosztów: normy `ΔW = (alpha/r)·BA` w kolejnych warstwach, rozkład VRAM (wagi / stany AdamW / aktywacje) | J2.4, B2.4 | 18.01 | Wykresy i wyjaśnienie, czemu spadek VRAM jest mniejszy niż spadek % parametrów |
| E2.1 | wspólnie | Aktualizacja `run_colab.ipynb` (baseline vs LoRA), raport/prezentacja, code freeze, próba generalna (21.01) | B2.5, J2.5 | **22.01 EOD** | Tag `v2.0-lora-final` na `main`, komplet materiałów przesłany |

---

## 10. Zasady pracy

1. **Code review:** nic nie trafia do `main` bez przeglądu drugiej osoby i krótkiego opisu działania kodu w PR.
2. **Synchronizacja:** wtorek i piątek, 30 minut. Przegląd tablicy zadań oraz krzywych i tabel w W&B.
3. **Próby generalne z zamianą ról** (17.12 przed P3, 21.01 przed P4):
   - Janek omawia pipeline danych YouTube, funkcję straty, metryki i wnioski z analizy błędów `cleanlab`,
   - Bruno omawia matematykę `LoRALinear` (inicjalizacja `A` i `B`, skalowanie `alpha/r`, `merged_weight`), działanie `torchtest2` oraz rozkład VRAM i FLOPs.
4. **Reprodukowalność:** stałe seedy, wszystkie parametry w konfiguracjach Hydra, jedno polecenie na run.
5. **Środowiska i sekrety** (klucz W&B, dane Kaggle) trzymamy tylko lokalnie, nigdy w repozytorium ani w ProtonDrive.

---

## 11. Obrona i ocena

Projekt jest oceniany na 30 pkt. **Najważniejsze jest rozumienie własnego rozwiązania i umiejętność wyjaśnienia go ustnie**, nie sam działający kod.

| Spotkanie | Pkt | Z tego rozumienie (obrona ustna) |
|---|---:|---:|
| P3: baseline | 10 | 4 |
| P4: rozszerzenie i analiza | 20 | 8 |

**Próg:** żeby dostać więcej niż 15/30, trzeba mieć co najmniej 7/12 pkt łącznie za rozumienie **oraz** co najmniej 4/8 pkt za rozumienie rozszerzenia. Kary: brak przekazania projektu na czas to −20 % punktów, a każdy dzień zwłoki względem dnia obrony to −1 pkt.

### Pytania, na które oboje musimy umieć odpowiedzieć

**P3 (baseline):**
- Jakie są wejścia i wyjścia modelu? Jak zmieniają się kształty (`[B,T]` → `[B,T,768]` → `[B,768]` → `[B,6]`)?
- Co robi backbone, a co głowica? Dlaczego pooling ostatniego nie-paddingowego tokenu?
- Dlaczego `CrossEntropyLoss`? Które parametry są trenowane (i ile)?
- Co znaczą accuracy i macro-F1 i czym się różnią? Jak czytamy macierz pomyłek?
- Dlaczego split jest po `video_id` i wspólny dla wszystkich eksperymentów?

**P4 (LoRA):**
- Skąd się bierze wzór `h = Wx + (alpha/r)·BAx` i dlaczego `B = 0` na starcie?
- Co robi `alpha/r` i dlaczego trzymamy `alpha/r = 2`?
- Jak wygląda rachunek parametrów `r·(d_in + d_out)` i ile to dla `q+v` w OPT-125M?
- Dlaczego VRAM spada mniej niż liczba parametrów?
- Co robi `merged_weight()` i dlaczego po scaleniu nie ma narzutu FLOPs?
- Czemu `q_proj`/`v_proj`? Co pokazała ablacja rang i warstw? Co, jeśli LoRA nie pobiła baseline'u?

### Mapowanie rubryki na nasze elementy

| Kryterium rubryki | Gdzie to realizujemy |
|---|---|
| Działający baseline end-to-end (3) | E0, B1.1–B1.5 |
| Poprawna ewaluacja i metryki (2) | `LitVideoClassifier`, `eval.py`, sekcja 6 |
| Kod uruchamialny i uporządkowany (1) | `ruff`, `pytest`, Hydra |
| Poprawna implementacja rozszerzenia (5) | J2.1–J2.3 (`LoRALinear` + testy + parytet z `peft`) |
| Baseline vs rozszerzenie + ablacje (3) | E1, E2, E3 |
| Analiza wyników i błędów (2) | B2.5, `cleanlab` |
| Analiza kosztu: parametry, czas, VRAM (1) | B1.4, J2.5, `fvcore` |
| Reprodukowalność i demonstracja (1) | `run_colab.ipynb`, seedy, `splits.json` |

---

## 12. Uruchomienie

Środowisko zarządzane przez `uv`, trzymane lokalnie (repozytorium **nie** leży w ProtonDrive, patrz `REPOZYTORIUM.md` w folderze projektu).

```bash
git clone https://github.com/BrunoSienkiewicz/lora-from-scratch.git
cd lora-from-scratch
uv sync --extra dev
uv run pytest                                      # na starcie czerwone (NotImplementedError)
uv run jupyter lab notebooks/00_wstepne_zrozumienie.ipynb
```

Po implementacji (docelowe polecenia):

```bash
uv run python scripts/train.py experiment=e0_baseline                      # baseline
uv run python scripts/train.py -m lora.rank=2,4,8,16 lora.targets=q_v      # seria rang
uv run python scripts/train.py trainer.fast_dev_run=true                   # szybki test pipeline'u
```

Wymagany jest klucz W&B (`wandb login`) i dostęp do danych Kaggle (`kaggle.json`). Bez nich użyj `logger=none` / `WANDB_MODE=offline`.

---

## 13. Decyzje, ryzyka i sprawy do potwierdzenia

### Decyzje podjęte (można zmienić, ale wspólnie)

- Zadanie i zbiór: kategoryzacja YouTube, 6 klas, 12 000 rekordów (sekcja 3).
- Baseline: head-only. Pełny fine-tuning jest jednym dodatkowym runem kontrolnym.
- `peft` tylko w testach jako wyrocznia, nie w logice treningu.
- Metryki: accuracy + macro-F1. Koszt: parametry, czas, VRAM, FLOPs.
- Rangi: `{2, 4, 8, 16}` z `alpha = 2r`. Warstwy: `q`, `q+v`, `q+k+v+o`, opcjonalnie `+FFN`.
- Sprzęt: 1 GPU T4 (Colab/Kaggle), pomiar VRAM na GPU.
- Pooling: ostatni nie-paddingowy token (`attention_mask.sum - 1`), padding z prawej.

### Ryzyka do sprawdzenia w E0.1 / B1.1 (przed implementacją reszty)

- [ ] **Dostępność danych na klasę.** Czy po deduplikacji po `video_id` w US/GB/CA jest co najmniej 2 000 unikalnych filmów w każdej z 6 kategorii? W starszych dumpach *Trending* kategorie Gaming i Sports bywają małe. Jeśli nie, wybieramy: mniej klas, mniejszy rozmiar zbioru albo zbiór niezrównoważony (wtedy macro-F1 staje się głównym wskaźnikiem).
- [ ] **Pobieranie danych.** Dane są na Kaggle i wymagają `kaggle.json`. Ustalić, jak Colab dostaje dane (API Kaggle w notebooku albo przetworzony plik w Drive/HF) i kto go przygotowuje.
- [ ] **`torchtest2` (v0.1.0, status alfa, Python ≥ 3.10, PyTorch ≥ 2.1, `pip install torchtest2`, [repo](https://github.com/Hrehory/torchtest2)).** API: `check_model(...)` zwraca `Report`, a `assert_model_ok(model, x, y, loss_fn=..., optimizer=..., output_shape=..., frozen=...)` nadaje się do `pytest`. Sprawdzenia m.in.: `frozen_params`, `params_change`, `gradient_flow`, `output_finite`, `overfit_batch`. Dochodzą asercje na tensorach (`assert_never_nan`, `assert_never_inf`, `assert_shape`). Biblioteka jest w wersji alfa, więc w E0.1 zweryfikować, że działa na `OPTClassifier` (szczególnie argument `frozen` i `forward` ze słownikiem wejść, być może przez `forward_fn`). Plan B: własne asercje (porównanie `state_dict` przed/po kroku, `torch.isfinite`).
- [ ] **Wersje bibliotek.** Zgodność `lightning`, `hydra-core`, `wandb`, `fvcore` i `torch` sprawdzić jednym `uv sync` zanim ktokolwiek zacznie kodować.
- [ ] **`cleanlab` wymaga prawdopodobieństw spoza zbioru uczącego.** Używamy predykcji na walidacji i teście (nie na train), więc dla analizy szumu etykiet to wystarcza, ale nie jest to pełna walidacja krzyżowa.

### Do potwierdzenia z prowadzącym (jeśli będzie okazja)

- Czy zadanie i zbiór (YouTube zamiast sugerowanych SST-2 / AG News) są akceptowalne? Opis tematu mówi o „kilku–kilkunastu tysiącach krótkich tekstów", więc 12 000 mieści się w zakresie.
- Czy używanie `peft` w testach jednostkowych jako wyroczni jest w porządku?
- Czy oczekiwany jest pomiar VRAM na GPU, czy wystarczy CPU?

---

## 14. Literatura

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021), arXiv:2106.09685
- Zhang et al., *OPT: Open Pre-trained Transformer Language Models* (2022), arXiv:2205.01068
