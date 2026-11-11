# Fuzzy Rule-Based Network — Reproduction and Extension

> Bulanık Mantık dersi dönem projesi. Durum: **Checkpoint 3 sürüyor**: FN çekirdeği, NN baseline, Adam / LM / LM trust-region eğitimi ve 5-katlı CV hazır; deney betikleri sırada. [Checkpoint 1](docs/checkpoint1_makale_analizi.md) · [Checkpoint 2](docs/checkpoint2_uygulama_plani.md)

## Project Title
Fuzzy Rule-Based Network (FRBN): Yorumlanabilir ve Modüler Bulanık Ağın Yeniden Gerçekleştirilmesi ve Geliştirilmesi

## Student Information
| | |
|---|---|
| Ad Soyad | Turan Çaldıran |
| Öğrenci No | 1306220018 |
| Ders | Bulanık Mantık — Dönem Projesi |
| Atanan makale | No. 44 |

## Original Paper
H.-I. Lukács, Á. Smidla, T. Fischl, and J. Botzheim, "Fuzzy rule-based network: a fuzzy logic-based interpretable and modular machine learning model," *Scientific Reports*, vol. 16, Art. no. 34429, 2026, doi: [10.1038/s41598-026-63687-6](https://doi.org/10.1038/s41598-026-63687-6).
Official code: <https://github.com/hunorlukacs/Fuzzy-Network>

## Problem Description
Derin sinir ağları doğru ama yorumlanamaz ("kara kutu"). Klasik bulanık sistemler ve ANFIS ise yorumlanabilir ama derin yapılar kurmaya uygun değil. FRBN, nöron katmanları yerine **bulanık kural katmanları** kullanarak bu iki yaklaşımı birleştiren "gri kutu" bir model öneriyor. Ayrıntılar: [docs/checkpoint1_makale_analizi.md](docs/checkpoint1_makale_analizi.md)

## Method
- Katman `[M, N, K]`: M giriş, N çıkış, K kural; öncüller paylaşımlı (MIMO).
- Üyelik fonksiyonları: trapezoid (a,b,c,d) veya Gauss (μ,σ).
- Kural aktivasyonu: min t-norm.
- Durulaştırma: Mamdani + Center of Sums (trapezoid) / ağırlıklı merkez ortalaması (Gauss).
- Eğitim: Adam, Levenberg–Marquardt (trust region), BEA/BMA.

## Fuzzy System Architecture
`x → [Fuzzy Layer 1] → … → [Fuzzy Layer L] → y` _(diyagram eklenecek: `results/figures/architecture.png`)_

| Modül | Görev |
|---|---|
| [src/membership.py](src/membership.py) | Trapezoid (Denk. 2–3) ve Gauss (Denk. 7) üyelik fonksiyonları |
| [src/fuzzy_layer.py](src/fuzzy_layer.py) | `FuzzyLayer [M,N,K]`: min t-norm, Center of Sums (Denk. 5–6), ağırlıklı ortalama (Denk. 9) |
| [src/fuzzy_network.py](src/fuzzy_network.py) | `FuzzyNetwork`: katmanların zincirlenmesi (Denk. 10), parametre sayımı, kuralların yazdırılması |
| [src/data.py](src/data.py) | sinc / trigonometrik veri üreticileri, 5-katlı bölme |
| [src/nn_baseline.py](src/nn_baseline.py) | Karşılaştırma için tek gizli katmanlı MLP (1-20-1 = 61 parametre) |
| [src/optim_lm.py](src/optim_lm.py) | Levenberg–Marquardt: standart ve trust-region sönümleme (Denk. 15–19) |
| [src/train.py](src/train.py) | Eğitim döngüsü (Adam / LM / LM-TR), early stopping, 5-katlı CV, özet istatistikler |

```python
from src.fuzzy_network import FuzzyNetwork
net = FuzzyNetwork([[3, 1, 7]], ante="trap", cons="trap", seed=0)
print(net.summary()); print("\n".join(net.rules()))
```

## Dataset
- **sinc:** sentetik, 20.000 nokta (kodla üretilir).
- **Trigonometrik:** y = sin(2πx₂)·cos(2πx₃), x₁ gürültü, 8000 örnek (kodla üretilir).
- MEMS endüstriyel veri seti bu projenin kapsamı dışındadır (gerekçe raporda).

## Installation
```bash
git clone <repo-url>
cd fuzzy-logic-project
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Requirements
Python 3.14.5, PyTorch 2.15.0 (CPU). Sabitlenmiş sürümler: [requirements.txt](requirements.txt).

## How to Run
```bash
python -m src.data                         # veri setlerini üretip özetini yazdırır
python -m pytest -q                        # testleri çalıştırır
python -m experiments.baseline.e1_sinc     # E1: sinc, FN vs NN (~25 dk, CPU)
python -m experiments.baseline.e2_trig     # E2: trigonometrik veri, [3,1,7] FN
```
Her betik `--quick` ile birkaç saniyelik bir deneme modunda da çalışır. Çıktılar `results/tables/` (CSV + Markdown özet) ve `results/figures/` (PNG) altına yazılır.

## Reproduction Experiments
| Deney | Makale sonucu | Bizim sonucumuz |
|---|---|---|
| sinc, FN Gauss LM-TR (val. MSE, en iyi fold) | 6.20e-8 | — |
| sinc, NN tanh BMA-LM-TR (val. MSE, ort.) | 2.33e-7 | — |
| Trig [3,1,7], LM-TR (val. loss, en iyi fold) | 0.001259 | — |

## Proposed Extension
_(Checkpoint 4'te belirlenecek; adaylar analiz belgesinde.)_

## Experimental Results
_(Checkpoint 5'te eklenecek.)_

## Limitations
_(Raporla birlikte doldurulacak.)_

## References
Bkz. [references/README.md](references/README.md).

## Code Provenance and AI Usage
- `src/` ve `experiments/` altındaki kod bu proje kapsamında yazılmıştır; resmi depodan kod kopyalanmamıştır (resmi depo yalnızca makalede yazmayan ayrıntılar için referans olarak okunmuştur). Dışarıdan bir parça alınırsa dosya başında kaynak gösterilerek işaretlenecektir.
- Kodun ve belgelerin hazırlanmasında üretken yapay zekâ desteği kullanılmıştır; ayrıntılar raporun "Generative AI Usage Disclosure" bölümünde beyan edilecektir.
