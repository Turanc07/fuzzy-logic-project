# Data

Bu projede dosya olarak saklanan bir veri seti yoktur: kullanılan iki veri seti de sentetiktir ve [src/data.py](../src/data.py) tarafından sabit tohumlarla her çalıştırmada aynı şekilde üretilir.

| Veri seti | Üretim | Boyut |
|---|---|---|
| sinc | `make_sinc(n=20_000)`: x ∈ [-1, 1] eşit aralıklı, y = sin(10πx) / (10πx) | 20.000 × (1 → 1) |
| Trigonometrik | `make_trig(n=8000, seed=0)`: x₁, x₂, x₃ ~ U[0, 1] (x₁ gürültü), y = (sin(2πx₂)·cos(2πx₃) + 1) / 2 | 8.000 × (3 → 1) |
| Korelasyonlu gürültü (ablation) | `make_trig(nuisance="correlated", noise_std=s)`: x₁ = 0.3·x₂ + ε | 8.000 × (3 → 1) |

Özet görmek için: `python -m src.data`

Makaledeki endüstriyel MEMS veri seti (yaklaşık 800 bin kayıt, anonimleştirilmiş) bu projenin kapsamı dışındadır.
