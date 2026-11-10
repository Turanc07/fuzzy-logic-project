# Fuzzy Rule-Based Network — Reproduction and Extension

> Bulanık Mantık dersi dönem projesi. Durum: **Checkpoint 1 (makale analizi) tamamlandı**, uygulama henüz başlamadı.

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
pip install -r requirements.txt
```

## Requirements
Bkz. [requirements.txt](requirements.txt). Test edildikten sonra kesin sürümler sabitlenecek.

## How to Run
_(Checkpoint 3'te eklenecek.)_

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
- `src/` altındaki kod aksi belirtilmedikçe öğrenci tarafından yazılmıştır. Resmi depodan alınan her parça dosya başında kaynak gösterilerek işaretlenecektir.
- Üretken yapay zekâ araçlarının kullanımı raporun "Generative AI Usage Disclosure" bölümünde belirtilecektir.
