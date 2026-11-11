# E2 — trigonometrik veri sonuçları

y, (sin·cos + 1)/2 ile [0, 1]'e normalize edildi (makale y'nin normalize olduğunu söylüyor).

## Makale protokolü (sabit başlatma, seed 42)

| Konfigürasyon | Val. MSE ort. ± std (bizim) | En iyi fold (bizim) | Yakınsamayan fold | Makale ort. ± std | Makale en iyi | Süre [s] | Param. |
|---|---|---|---|---|---|---|---|
| FN trap adam | 1.99e-02 ± 6.51e-03 | 9.48e-03 | 0/5 | 5.38e-02 ± 1.30e-02 | 3.83e-02 | 5.4 | 112 |
| FN trap lm | 2.77e-02 ± 1.53e-02 | 7.05e-03 | 0/5 | 1.06e-02 ± 1.43e-02 | 3.03e-03 | 14.8 | 112 |
| FN trap lm_tr | 2.78e-02 ± 1.01e-02 | 1.49e-02 | 0/5 | 1.09e-02 ± 1.81e-02 | 1.26e-03 | 15.4 | 112 |

## Değişken başlatma (her fold'da farklı seed)

| Konfigürasyon | Val. MSE ort. ± std (bizim) | En iyi fold (bizim) | Yakınsamayan fold | Makale ort. ± std | Makale en iyi | Süre [s] | Param. |
|---|---|---|---|---|---|---|---|
| FN trap adam | 1.56e-02 ± 7.88e-03 | 4.40e-03 | 0/5 | 5.38e-02 ± 1.30e-02 | 3.83e-02 | 5.4 | 112 |
| FN trap lm | 2.98e-02 ± 1.63e-02 | 7.61e-03 | 0/5 | 1.06e-02 ± 1.43e-02 | 3.03e-03 | 14.8 | 112 |
| FN trap lm_tr | 3.27e-02 ± 3.48e-02 | 6.60e-03 | 1/5 | 1.09e-02 ± 1.81e-02 | 1.26e-03 | 13.2 | 112 |

## Gürültü özelliği x1 (tüm veriyle eğitilen LM-TR modeli)

Eğitim MSE: 2.768e-02

| Kural | x1 MF'sinin [0,1] üzerindeki ortalama üyeliği |
|---|---|
| R0 | 0.567 |
| R1 | 1.000 |
| R2 | 0.000 |
| R3 | 0.939 |
| R4 | 0.579 |
| R5 | 1.000 |
| R6 | 0.882 |

Kurallar:
```
R0: IF x1 is trap(-1.161, 1.769, 2.167, 2.181) AND x2 is trap(-0.254, 0.134, 0.803, 1.381) AND x3 is trap(0.155, 1.427, 1.804, 1.822) THEN y1 is trap(-0.138, 0.023, 0.024, 0.027)
R1: IF x1 is trap(-0.330, -0.026, 1.223, 1.608) AND x2 is trap(0.845, 0.969, 1.226, 1.655) AND x3 is trap(-0.191, -0.109, 1.093, 1.271) THEN y1 is trap(-1.976, -1.657, 2.639, 2.938)
R2: IF x1 is trap(-0.684, -0.402, -0.249, -0.031) AND x2 is trap(-0.982, -0.774, 0.642, 1.855) AND x3 is trap(-0.034, 0.738, 1.427, 1.658) THEN y1 is trap(-0.615, -0.308, 1.385, 1.482)
R3: IF x1 is trap(-0.290, 0.240, 0.904, 1.754) AND x2 is trap(-0.259, 0.631, 1.131, 1.839) AND x3 is trap(0.646, 2.603, 2.603, 2.603) THEN y1 is trap(0.720, 0.720, 0.720, 1.682)
R4: IF x1 is trap(-0.985, -0.817, -0.767, 2.240) AND x2 is trap(-1.126, 0.398, 1.121, 1.731) AND x3 is trap(-0.724, 0.171, 0.974, 1.674) THEN y1 is trap(0.781, 0.783, 0.784, 0.823)
R5: IF x1 is trap(-0.524, -0.017, 1.053, 1.415) AND x2 is trap(0.440, 0.704, 1.011, 1.745) AND x3 is trap(-0.395, -0.395, 1.005, 1.849) THEN y1 is trap(-1.170, -1.169, -0.843, 2.273)
R6: IF x1 is trap(-0.988, -0.850, -0.240, 6.032) AND x2 is trap(0.455, 0.725, 1.620, 1.746) AND x3 is trap(-0.523, -0.300, 0.659, 0.899) THEN y1 is trap(-2.080, 1.873, 2.408, 2.972)
```