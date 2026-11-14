# Checkpoint 5 — Baseline ve önerilen yöntemin karşılaştırması

`*` = p < 0.05. "Daha iyi çift": önerilen yöntemin random'dan daha düşük hata verdiği eşleştirilmiş çiftlerin oranı.

## Özet: Baseline ve önerilen yöntem (yönerge §6.2 formatı)

| Performans ölçütü | Baseline (random) | Önerilen (fcm) | Değişim |
|---|---|---|---|
| sinc LM-TR yakınsama oranı (P1) | 45% | 100% | +55 puan |
| sinc LM-TR medyan val. MSE (P1) | 1.98e-03 | 1.05e-07 | 18821× daha düşük |
| sinc LM-TR toplam süre [s] (P1) | 6.8 | 8.0 | +19% |
| trig LM-TR ort. val. MSE (P3) | 3.32e-02 | 2.25e-02 | 1.5× daha düşük |
| trig LM-TR RMSE (P3) | 0.1821 | 0.1501 | -18% |
| trig başlangıçta ölü kural (ort., P3) | 0.40 | 0.00 | — |
| trig LM-TR toplam süre [s] (P3) | 7.9 | 10.0 | +27% |

## P1 — Yakınsama oranı (sinc, Gauss `[1,1,20]`, fold 0, 20 başlatma)

Başarı: val. MSE < 1e-05. Testler random'a karşı, aynı tohumlarla eşleştirilmiş.

| Optimizer | Başlatma | Yakınsayan | Medyan val. MSE | En iyi | En kötü | Başlangıç MSE (medyan) | Fisher p | Wilcoxon p | Daha iyi olduğu çift |
|---|---|---|---|---|---|---|---|---|---|
| LM-TR | random (baseline) | **9/20** | 1.98e-03 | 1.79e-08 | 1.98e-02 | 6.42e-02 | — | — | — |
| LM-TR | lhs (kontrol) | **20/20** | 1.58e-08 | 1.54e-08 | 2.97e-08 | 3.18e-02 | 0.000 * | 0.000 * | 95% |
| LM-TR | fcm (önerilen) | **20/20** | 1.05e-07 | 1.47e-08 | 9.54e-06 | 1.55e-02 | 0.000 * | 0.000 * | 80% |
| LM-TR | fcm_xy (önerilen varyant) | **17/20** | 2.04e-07 | 2.81e-08 | 1.70e-04 | 1.11e-02 | 0.019 * | 0.002 * | 75% |
| LM | random (baseline) | **13/20** | 7.04e-07 | 2.90e-08 | 8.40e-03 | 6.42e-02 | — | — | — |
| LM | lhs (kontrol) | **20/20** | 1.44e-08 | 1.44e-08 | 1.44e-08 | 3.18e-02 | 0.008 * | 0.000 * | 100% |
| LM | fcm (önerilen) | **19/20** | 8.81e-08 | 9.63e-09 | 1.26e-05 | 1.55e-02 | 0.044 * | 0.002 * | 75% |
| LM | fcm_xy (önerilen varyant) | **17/20** | 2.28e-07 | 1.36e-08 | 1.57e-04 | 1.11e-02 | 0.273 | 0.027 * | 65% |
| Adam | random (baseline) | **2/20** | 1.05e-04 | 4.93e-06 | 8.22e-04 | 6.42e-02 | — | — | — |
| Adam | lhs (kontrol) | **0/20** | 2.05e-05 | 1.79e-05 | 2.41e-05 | 3.18e-02 | 0.487 | 0.003 * | 75% |
| Adam | fcm (önerilen) | **10/20** | 1.01e-05 | 3.37e-06 | 3.92e-04 | 1.55e-02 | 0.014 * | 0.019 * | 85% |
| Adam | fcm_xy (önerilen varyant) | **3/20** | 8.67e-05 | 3.50e-06 | 1.98e-04 | 1.11e-02 | 1.000 | 0.312 | 65% |

## P2 — sinc, 5-katlı CV (her fold'da farklı başlatma)

Not: 5 eşleştirilmiş çiftle Wilcoxon testinin ulaşabileceği en küçük p değeri 0.0625; bu tablo betimleyicidir, anlamlılık P1'de test edilir.

| Model | Optimizer | Başlatma | Val. MSE ort. ± std | Medyan | En iyi | Yakınsayan | Ölü kural (başta) | Başlatma süresi [s] | Eğitim süresi [s] | Wilcoxon p | Daha iyi çift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gauss | Adam | random (baseline) | 2.81e-04 ± 3.78e-04 | 6.98e-05 | 2.00e-05 | 0/5 | 0.00 | 0.000 | 1.7 | — | — |
| gauss | Adam | lhs (kontrol) | 1.43e-05 ± 4.24e-06 | 1.19e-05 | 9.22e-06 | 1/5 | 0.00 | 0.003 | 1.8 | 0.062 | 100% |
| gauss | Adam | fcm (önerilen) | 3.10e-05 ± 3.86e-05 | 1.31e-05 | 4.42e-06 | 1/5 | 0.00 | 2.082 | 1.8 | 0.125 | 80% |
| gauss | Adam | fcm_xy (önerilen varyant) | 1.13e-04 ± 6.50e-05 | 1.14e-04 | 9.75e-06 | 1/5 | 0.00 | 2.201 | 1.9 | 0.625 | 60% |
| gauss | LM | random (baseline) | 1.43e-06 ± 1.23e-06 | 1.24e-06 | 9.40e-08 | 5/5 | 0.00 | 0.000 | 5.7 | — | — |
| gauss | LM | lhs (kontrol) | 1.49e-07 ± 1.40e-07 | 9.11e-08 | 1.44e-08 | 5/5 | 0.00 | 0.003 | 6.0 | 0.125 | 80% |
| gauss | LM | fcm (önerilen) | 1.36e-07 ± 7.40e-08 | 1.00e-07 | 4.65e-08 | 5/5 | 0.00 | 2.078 | 6.0 | 0.062 | 100% |
| gauss | LM | fcm_xy (önerilen varyant) | 6.80e-05 ± 5.83e-05 | 8.48e-05 | 4.83e-07 | 2/5 | 0.00 | 2.184 | 5.9 | 0.312 | 40% |
| gauss | LM-TR | random (baseline) | 3.24e-03 ± 6.18e-03 | 2.34e-07 | 1.05e-07 | 3/5 | 0.00 | 0.000 | 5.2 | — | — |
| gauss | LM-TR | lhs (kontrol) | 6.54e-08 ± 4.25e-08 | 3.59e-08 | 2.89e-08 | 5/5 | 0.00 | 0.003 | 5.5 | 0.062 | 100% |
| gauss | LM-TR | fcm (önerilen) | 1.12e-07 ± 8.40e-08 | 7.31e-08 | 2.51e-08 | 5/5 | 0.00 | 2.088 | 5.5 | 0.062 | 100% |
| gauss | LM-TR | fcm_xy (önerilen varyant) | 7.64e-05 ± 9.55e-05 | 1.70e-06 | 1.88e-07 | 3/5 | 0.00 | 2.209 | 5.6 | 1.000 | 60% |
| trap | Adam | random (baseline) | 1.91e-03 ± 7.11e-04 | 1.86e-03 | 1.02e-03 | 0/5 | 0.00 | 0.000 | 4.9 | — | — |
| trap | Adam | lhs (kontrol) | 2.98e-05 ± 1.02e-05 | 2.45e-05 | 2.32e-05 | 0/5 | 0.00 | 0.003 | 4.9 | 0.062 | 100% |
| trap | Adam | fcm (önerilen) | 8.56e-05 ± 6.38e-05 | 9.08e-05 | 2.32e-05 | 0/5 | 0.00 | 2.061 | 4.9 | 0.062 | 100% |
| trap | Adam | fcm_xy (önerilen varyant) | 1.58e-04 ± 5.34e-05 | 1.66e-04 | 8.64e-05 | 0/5 | 0.00 | 2.186 | 4.9 | 0.062 | 100% |
| trap | LM | random (baseline) | 1.29e-03 ± 5.88e-04 | 1.35e-03 | 3.65e-04 | 0/5 | 0.00 | 0.000 | 16.2 | — | — |
| trap | LM | lhs (kontrol) | 5.53e-05 ± 1.71e-05 | 4.79e-05 | 3.92e-05 | 0/5 | 0.00 | 0.003 | 15.0 | 0.062 | 100% |
| trap | LM | fcm (önerilen) | 4.03e-05 ± 5.67e-06 | 4.18e-05 | 2.96e-05 | 0/5 | 0.00 | 2.581 | 22.0 | 0.062 | 100% |
| trap | LM | fcm_xy (önerilen varyant) | 2.61e-04 ± 7.24e-05 | 2.70e-04 | 1.59e-04 | 0/5 | 0.00 | 3.143 | 24.1 | 0.062 | 100% |
| trap | LM-TR | random (baseline) | 9.02e-04 ± 6.08e-04 | 7.55e-04 | 2.63e-05 | 0/5 | 0.00 | 0.000 | 22.2 | — | — |
| trap | LM-TR | lhs (kontrol) | 1.28e-03 ± 2.40e-03 | 8.38e-05 | 5.32e-05 | 0/5 | 0.00 | 0.003 | 16.5 | 0.625 | 80% |
| trap | LM-TR | fcm (önerilen) | 8.76e-05 ± 3.54e-05 | 8.76e-05 | 4.70e-05 | 0/5 | 0.00 | 1.916 | 12.4 | 0.125 | 80% |
| trap | LM-TR | fcm_xy (önerilen varyant) | 7.99e-04 ± 5.17e-04 | 8.36e-04 | 9.40e-05 | 0/5 | 0.00 | 2.015 | 15.2 | 1.000 | 60% |

## P3/P4 — Trigonometrik veri, 5-katlı CV × 2 başlatma (10 eşleştirilmiş çift)

Yorumlanabilirlik sütunları: başlangıçtaki / eğitim sonundaki ölü kural sayısı, hiçbir kuralın ateşlenmediği doğrulama örneği oranı, ortalama kapsama oranı ρ (Denk. 14; 1 = global kural), gürültü özelliği x₁'i tamamen yok sayan kural sayısı.

| Model | Optimizer | Başlatma | Val. MSE ort. ± std | Medyan | En iyi | Ölü kural (başta) | Ölü kural (sonda) | Kapsanmayan örnek | ρ ort. | x₁'i yok sayan kural | Başlatma süresi [s] | Eğitim süresi [s] | Wilcoxon p | Daha iyi çift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trap | Adam | random (baseline) | 1.44e-02 ± 6.47e-03 | 1.46e-02 | 4.40e-03 | 0.40 | 0.90 | 0.01 | 0.47 | 4.30 | 0.000 | 2.8 | — | — |
| trap | Adam | lhs (kontrol) | 9.61e-03 ± 1.27e-02 | 5.54e-03 | 3.17e-03 | 0.00 | 0.30 | 0.01 | 0.52 | 5.30 | 0.003 | 2.8 | 0.037 * | 80% |
| trap | Adam | fcm (önerilen) | 1.75e-02 ± 1.92e-02 | 5.57e-03 | 3.56e-03 | 0.00 | 0.30 | 0.03 | 0.45 | 5.00 | 0.409 | 2.9 | 0.492 | 60% |
| trap | Adam | fcm_xy (önerilen varyant) | 1.23e-02 ± 1.79e-02 | 3.88e-03 | 1.21e-03 | 0.00 | 0.20 | 0.02 | 0.50 | 5.20 | 0.426 | 2.8 | 0.193 | 80% |
| trap | LM | random (baseline) | 2.33e-02 ± 1.50e-02 | 2.45e-02 | 5.53e-03 | 0.40 | 1.30 | 0.01 | 0.47 | 3.30 | 0.000 | 8.5 | — | — |
| trap | LM | lhs (kontrol) | 6.66e-03 ± 9.95e-04 | 6.71e-03 | 5.03e-03 | 0.00 | 0.10 | 0.00 | 0.67 | 5.20 | 0.003 | 8.4 | 0.006 * | 90% |
| trap | LM | fcm (önerilen) | 6.20e-03 ± 1.07e-02 | 1.61e-03 | 3.90e-04 | 0.00 | 0.10 | 0.04 | 0.49 | 4.60 | 0.413 | 8.9 | 0.014 * | 90% |
| trap | LM | fcm_xy (önerilen varyant) | 7.11e-03 ± 1.08e-02 | 2.64e-03 | 3.12e-04 | 0.00 | 0.30 | 0.01 | 0.58 | 5.90 | 0.436 | 8.3 | 0.010 * | 90% |
| trap | LM-TR | random (baseline) | 3.32e-02 ± 3.82e-02 | 8.77e-03 | 5.50e-03 | 0.40 | 1.00 | 0.04 | 0.51 | 3.00 | 0.000 | 7.9 | — | — |
| trap | LM-TR | lhs (kontrol) | 6.86e-03 ± 1.32e-03 | 6.89e-03 | 5.07e-03 | 0.00 | 0.00 | 0.00 | 0.70 | 5.30 | 0.003 | 7.7 | 0.049 * | 70% |
| trap | LM-TR | fcm (önerilen) | 2.25e-02 ± 2.46e-02 | 1.12e-02 | 7.76e-04 | 0.00 | 0.00 | 0.06 | 0.50 | 2.60 | 0.427 | 9.6 | 0.492 | 70% |
| trap | LM-TR | fcm_xy (önerilen varyant) | 8.00e-03 ± 1.83e-02 | 1.80e-03 | 6.52e-04 | 0.00 | 0.10 | 0.01 | 0.59 | 4.90 | 0.436 | 8.0 | 0.010 * | 90% |
