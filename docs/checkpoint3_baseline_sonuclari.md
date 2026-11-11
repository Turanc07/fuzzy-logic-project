# Checkpoint 3 — Baseline Sistem ve İlk Deney Sonuçları

**Öğrenci:** Turan Çaldıran (1306220018) · **Makale No:** 44 · Önceki adımlar: [Checkpoint 1](checkpoint1_makale_analizi.md) · [Checkpoint 2](checkpoint2_uygulama_plani.md)

Bu belge, referans yöntemin (FRBN) yeniden gerçekleştirilmesini ve makaledeki iki temel deneyin tekrarını raporlar. Tüm sayılar `results/tables/` altındaki CSV dosyalarından alınmıştır; grafikler `results/figures/` altındadır.

---

## 1. Çalışır durumdaki sistem

| Bileşen | Dosya | Durum |
|---|---|---|
| Üyelik fonksiyonları (trapezoid, Gauss) | `src/membership.py` | ✅ birim testli |
| Bulanık katman: min t-norm, Center of Sums, ağırlıklı ortalama | `src/fuzzy_layer.py` | ✅ CoS kapalı formülü sayısal integralle doğrulandı |
| Çok katmanlı ağ, parametre sayımı, kural çıktısı | `src/fuzzy_network.py` | ✅ |
| NN baseline (1-20-1 MLP) | `src/nn_baseline.py` | ✅ |
| Adam, LM, LM trust-region | `src/train.py`, `src/optim_lm.py` | ✅ |
| 5-katlı CV, iki başlatma protokolü, tablo/grafik üretimi | `src/experiment.py` | ✅ |
| Deney betikleri | `experiments/baseline/e1_sinc.py`, `e2_trig.py`, `e1b_init_sensitivity.py` | ✅ |

Testler: `python -m pytest -q` → 35/35 geçiyor.

## 2. Deney düzeni

- **Donanım/yazılım:** Windows 11, CPU; Python 3.14.5, PyTorch 2.15.0.
- **Bölme:** 5-katlı CV, karıştırılmış, `split_seed = 0`.
- **Eğitim:** 100 epoch; her epoch eğitim kümesinin 1/10'u büyüklüğünde 10 mini-batch'ten oluşuyor, yani toplam 1000 optimizer adımı. Adam lr = 0.01; LM γ₀ = 1.
- **Başlatma protokolleri:**
  - *paper*: Tüm fold'lar aynı başlangıç ağırlıklarıyla eğitiliyor (seed 42). Resmi notebook her konfigürasyondan önce `set_seed(42)` çağırıyor; bu protokol onu taklit ediyor.
  - *varied*: Her fold farklı başlangıç ağırlıklarıyla eğitiliyor (seed 100 + fold). Varyans bu durumda başlatma etkisini de içeriyor.
- **Ölçüt:** Doğrulama MSE (ortalama ± std, en iyi fold), eğitim süresi, parametre sayısı.

## 3. E1 — sinc fonksiyonu (makale Tablo 1)

Kaynak: `results/tables/e1_sinc_summary.md` · Şekiller: `e1_sinc_curves_fn.png`, `e1_sinc_curves_nn.png`, `e1_sinc_preds_fn.png`, `e1_sinc_preds_nn.png`

| Konfigürasyon | Bizim (*paper* protokolü) | Bizim (*varied*) | Makale |
|---|---|---|---|
| FN Gauss LM-TR | 3.90e-3 ± 3.0e-4 | 3.24e-3 ± 6.2e-3 (en iyi **1.05e-7**) | **8.07e-7** ± 5.8e-7 |
| FN Gauss LM | 5.05e-3 ± 1.4e-3 | **1.43e-6** ± 1.2e-6 (en iyi 9.4e-8) | 6.05e-3 ± 4.0e-3 |
| FN Gauss Adam | **6.44e-5** ± 8.5e-5 | 2.81e-4 ± 3.8e-4 | 3.35e-4 ± 2.6e-4 |
| FN Trap LM | 1.00e-3 ± 5.4e-4 | 1.29e-3 ± 5.9e-4 | 6.65e-3 ± 5.9e-3 |
| FN Trap LM-TR | 1.41e-3 ± 9.3e-4 | 9.02e-4 ± 6.1e-4 | 8.33e-3 ± 6.6e-3 |
| FN Trap Adam | 1.86e-3 ± 3.8e-5 | 1.91e-3 ± 7.1e-4 | 2.24e-3 ± 8.3e-4 |
| NN tanh LM-TR | **1.68e-7** ± 7.5e-8 | 1.59e-7 ± 1.4e-7 | **2.82e-7** ± 3.8e-8 |
| NN tanh LM | 2.90e-7 ± 7.7e-8 | 2.01e-7 ± 8.0e-8 | 3.26e-5 ± 4.8e-5 |
| NN tanh Adam | 1.00e-2 ± 3.2e-4 | 1.34e-2 ± 6.2e-3 | 1.64e-2 ± 9.7e-3 |
| NN ReLU LM | 2.55e-3 ± 5.1e-4 | 1.99e-3 ± 3.4e-4 | 1.05e-2 ± 1.7e-2 |
| NN ReLU LM-TR | 3.16e-3 ± 1.5e-5 | 2.04e-3 ± 5.4e-4 | 2.77e-3 ± 4.7e-4 |
| NN ReLU Adam | 3.31e-2 ± 2.6e-4 | 1.14e-2 ± 1.1e-2 | 1.18e-2 ± 1.2e-2 |

**Başarıyla tekrarlananlar**
- **NN sonuçları makaleyle örtüşüyor:** tanh + LM-TR en iyi NN konfigürasyonu (1.7e-7, makalede 2.8e-7). ReLU tüm optimizer'larda 10⁻³ mertebesinde kalıyor. Bu, LM uygulamasının doğru olduğunu gösteriyor.
- **FN 10⁻⁷–10⁻⁸ seviyesine ulaşabiliyor:** Gauss + LM ile en iyi fold 9.4e-8, LM-TR ile 1.05e-7. Makalenin en iyi değeri 6.2e-8; E1b'de 1.8e-8'e kadar inen çalıştırmalar da var.
- Trapezoid FN, Gauss FN'den kötü. Makalede de durum böyle.
- FN Gauss + Adam ve trapezoid konfigürasyonlarında makaleyle aynı ya da daha iyi değerler elde edildi.

**Tekrarlanamayanlar / farklar**
- *paper* protokolünde FN Gauss + LM/LM-TR 5 fold'un hepsinde ~4e-3'te takıldı. Bunun nedeni bizim seed 42'mizin (rastgele sayı akışı makaledekinden farklı) kötü bir başlatmaya denk gelmesi. `e1_sinc_preds_fn.png` şeklinde bu model ana tepeyi ve uçları doğru öğreniyor, ama x ≈ ±0.1–0.3 aralığındaki yan dalgalanmaları **tamamen kaçırıyor**. O bölgeyi kapsayan bir kural olmadığı için gradyan da yok.
- Aynı konfigürasyon *varied* protokolde 5 fold'un 3–5'inde 10⁻⁶–10⁻⁷'ye iniyor. Yani makaledeki düşük std, **yalnızca tek bir (şanslı) başlatmanın** veri bölmesi varyansını yansıtıyor olabilir.

## 4. E1b — Başlatma duyarlılığı (yeni analiz)

Kaynak: `results/tables/e1b_init_summary.md` · Şekil: `e1b_init.png`. Veri bölmesi sabit (fold 0); yalnızca başlatma tohumu değişiyor (20 tohum).

| Konfigürasyon | Yakınsayan (val. MSE < 1e-5) | Medyan | En iyi | En kötü |
|---|---|---|---|---|
| FN Gauss LM-TR | **9/20** | 1.98e-3 | 1.79e-8 | 1.98e-2 |
| FN Gauss LM | **13/20** | 7.04e-7 | 2.90e-8 | 8.40e-3 |
| NN tanh LM-TR | **20/20** | 1.62e-7 | 1.00e-8 | 1.59e-6 |

- FN sonuçları **iki kümeli**: Çalıştırmalar ya ~10⁻⁷ seviyesine iniyor ya da ~10⁻³'te takılıyor; arası neredeyse yok. Aynı parametre bütçesindeki NN ise her başlatmada yakınsıyor.
- **Sonuç:** FN'nin yüksek doğruluğu gerçek, ancak **güvenilir değil**. Rastgele başlatma kuralları giriş uzayına kötü dağıtıyor ve kapsanmayan bölgeler (sıfır ateşleme → sıfır gradyan) optimizasyonla kurtarılamıyor. Makale bu duyarlılığı raporlamıyor.

## 5. E2 — Trigonometrik veri (makale Tablo 3)

Kaynak: `results/tables/e2_trig_summary.md` · Şekil: `e2_trig_curves.png`. Mimari `[3,1,7]`, trapezoid, 112 parametre.

| Optimizer | Bizim (*paper*) | Bizim (*varied*) | Makale |
|---|---|---|---|
| Adam | **1.99e-2** ± 6.5e-3 | 1.56e-2 ± 7.9e-3 | 5.38e-2 ± 1.3e-2 |
| LM | 2.77e-2 ± 1.5e-2 (en iyi 7.1e-3) | 2.98e-2 ± 1.6e-2 | **1.06e-2** ± 1.4e-2 (en iyi 3.0e-3) |
| LM-TR | 2.78e-2 ± 1.0e-2 (en iyi 1.5e-2) | 3.27e-2 ± 3.5e-2 (en iyi 6.6e-3) | **1.09e-2** ± 1.8e-2 (en iyi 1.3e-3) |

- Adam makaleden iyi, LM/LM-TR ise yaklaşık 2.5 kat kötü. Makaledeki "LM-TR en iyi" sıralaması bu veri setinde **tekrarlanamadı**.
- **Olası nedenler:**
  1. y'nin nasıl normalize edildiği makalede belirtilmemiş. Biz (y+1)/2 kullandık; farklı bir ölçek MSE'yi doğrudan ölçekler.
  2. Başlatma: tüm veriyle eğitilen modelde **R2 kuralı hiç ateşlenmiyor**. x₁ MF'si tamamen [0,1] dışında, yani 7 kuralın biri ölü.
  3. Makaledeki LM, harici bir kütüphaneyle farklı ayarlarla çalıştırılmış olabilir.
- **Gürültü özelliği x₁ (makale Şekil 12):** 7 kuralın 2'sinde x₁'in [0,1] üzerindeki ortalama üyeliği tam 1.000, 2'sinde ~0.9; yani bu kurallar x₁'i yok sayıyor. Kalanlarda ise x₁ hâlâ etkili. Makaledeki "model gürültü özelliğini tamamen filtreler" iddiası bizim modelde **kısmen** doğrulandı.

## 6. Genel değerlendirme

| Makale iddiası | Bizim sonucumuz |
|---|---|
| FN, sinc'te NN ile rekabetçi; en iyi fold ~6e-8 | ✅ Doğrulandı (en iyi 1.8e-8) |
| NN eğitimi FN'den daha kararlı | ✅ Doğrulandı ve güçlendirildi: NN 20/20, FN 9–13/20 |
| FN için en iyi optimizer LM-TR | ⚠️ Kısmen. sinc'te LM ve LM-TR benzer, trig'de Adam daha iyi |
| Trapezoid FN, Gauss FN'den kötü | ✅ Doğrulandı |
| Trig'de gürültü özelliği yok sayılır | ⚠️ Kısmen (7 kuraldan 2–4'ünde) |
| Düşük fold varyansı | ⚠️ Yalnızca sabit başlatmada; başlatma değişince varyans büyük |

## 7. Checkpoint 4'e etkisi

Sinc'teki kapsanmayan bölgeler, trig'deki ölü kural ve E1b'deki iki kümeli dağılım tek bir sorunu işaret ediyor: **rastgele başlatma kuralları giriş uzayına kötü yerleştiriyor.** Bu doğrudan test edilebilir bir katkı önerisi doğuruyor: kuralları giriş aralığına eşit (ızgara) ya da veri kantillerine göre yerleştiren **kapsama-bilinçli başlatma**. Baseline ile karşılaştırmada başarı ölçütü E1b'deki yakınsama oranı (9/20) ve E2'deki MSE olabilir. Ayrıntılar Checkpoint 4'te.

## 8. Tekrar üretme

```bash
python -m experiments.baseline.e1_sinc               # ~25 dk
python -m experiments.baseline.e1b_init_sensitivity  # ~10 dk
python -m experiments.baseline.e2_trig               # ~5 dk
```
