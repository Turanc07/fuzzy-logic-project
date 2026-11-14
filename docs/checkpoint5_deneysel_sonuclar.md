# Checkpoint 5 — Deneysel Sonuçlar: Baseline ve Önerilen Yöntem

**Öğrenci:** Turan Çaldıran (1306220018) · **Makale No:** 44 · Önceki adım: [Checkpoint 4](checkpoint4_katki_onerisi.md)

Karşılaştırılan yöntemler ([src/initialization.py](../src/initialization.py)):
- `random`: makaledeki rastgele başlatma (**baseline**)
- `fcm`: giriş uzayında Fuzzy C-Means (**önerilen**)
- `fcm_xy`: ürün uzayında FCM (önerilen varyant)
- `lhs`: kantil ızgarası, kümeleme yok (kontrol)

Mimari, parametre sayısı, optimizer, epoch, batch, veri bölmeleri ve tohumlar tüm yöntemlerde aynı; yalnızca ilk katmanın başlangıç değerleri farklı. Başlatma yalnızca eğitim fold'undan hesaplanıyor.

Tam tablolar: [results/tables/p_summary.md](../results/tables/p_summary.md) · Ham veriler: `results/tables/p1_convergence.csv`, `p2_sinc_cv_*.csv`, `p3_trig_cv_*.csv` · Şekiller: `results/figures/p1_convergence.png`, `p2_sinc_preds.png`, `p2_sinc_gauss_box.png`, `p2_sinc_trap_box.png`, `p3_trig_box.png`

İstatistik: Random'a karşı eşleştirilmiş (aynı tohum/fold) **Wilcoxon işaretli sıra testi** (log₁₀ MSE üzerinde, çift yönlü) ve yakınsama sayıları için **Fisher kesin testi**. `*` = p < 0.05.

---

## 1. Özet tablo (yönerge §6.2 formatı)

| Performans ölçütü | Baseline (random) | Önerilen (fcm) | Değişim |
|---|---|---|---|
| sinc, LM-TR: yakınsama oranı (P1, 20 başlatma) | 45% (9/20) | **100% (20/20)** | +55 puan (Fisher p < 0.001 *) |
| sinc, LM-TR: medyan val. MSE (P1) | 1.98e-3 | **1.05e-7** | ~19.000× düşük (Wilcoxon p < 0.001 *) |
| sinc, LM-TR: toplam süre [s] (başlatma + eğitim) | 6.8 | 8.0 | +19% |
| trig, LM: ortalama val. MSE (P3, 10 çift) | 2.33e-2 | **6.20e-3** | 3.8× düşük (p = 0.014 *) |
| trig, LM-TR: ortalama val. MSE (P3) | 3.32e-2 | 2.25e-2 | 1.5× düşük (p = 0.49, **anlamlı değil**) |
| trig, LM-TR: RMSE (P3) | 0.182 | 0.150 | −18% |
| trig: başlangıçta ölü kural (ort.) | 0.40 | **0.00** | tamamen ortadan kalktı |
| trig, LM-TR: toplam süre [s] | 7.9 | 10.0 | +27% |

Makaleyle karşılaştırma (trig, Tablo 3): Makalenin LM-TR ortalaması 1.09e-2. Önerilen başlatmalarla bizim LM ortalamamız 6.2e-3 (`fcm`) ve LM-TR ortalamamız 8.0e-3 (`fcm_xy`). Yani Checkpoint 3'te makalenin gerisinde kalan sonuçlarımız, katkıyla makale seviyesinin **altına** indi.

## 2. P1 — Yakınsama oranı (sinc, Gauss `[1,1,20]`, fold 0, 20 başlatma)

| Optimizer | random | lhs | **fcm** | fcm_xy |
|---|---|---|---|---|
| LM-TR | 9/20 | 20/20 * | **20/20 *** | 17/20 * |
| LM | 13/20 | 20/20 * | **19/20 *** | 17/20 |
| Adam | 2/20 | 0/20 | **10/20 *** | 3/20 |

(Yakınsama: val. MSE < 1e-5. Yıldız: Fisher p < 0.05.)

- **H1 doğrulandı.** Baseline'daki iki kümeli dağılım (ya ~1e-7 ya ~1e-3) önerilen başlatmayla ortadan kalkıyor. `p1_convergence.png` şeklinde fcm ve lhs noktalarının tamamı alt kümede.
- Adam'da da `fcm` yakınsamayı 2/20'den 10/20'ye çıkarıyor. `lhs` ise Adam ile hiç 1e-5'in altına inmiyor ama çok tutarlı (2e-5 civarı).
- Medyan başlangıç hatası random'da 6.4e-2, fcm'de 1.6e-2. Model eğitime kabaca doğru bir yaklaşımla başlıyor.

## 3. P2 — sinc, 5-katlı CV

5 eşleştirilmiş çiftte Wilcoxon'un en küçük p değeri 0.0625 olduğundan bu tablo betimleyicidir.

| Model | Optimizer | random | lhs | fcm | fcm_xy |
|---|---|---|---|---|---|
| Gauss | LM-TR | 3.24e-3 ± 6.2e-3 | 6.54e-8 ± 4.3e-8 | **1.12e-7 ± 8.4e-8** | 7.64e-5 ± 9.6e-5 |
| Gauss | LM | 1.43e-6 ± 1.2e-6 | 1.49e-7 ± 1.4e-7 | **1.36e-7 ± 7.4e-8** | 6.80e-5 ± 5.8e-5 |
| Gauss | Adam | 2.81e-4 ± 3.8e-4 | 1.43e-5 ± 4.2e-6 | **3.10e-5 ± 3.9e-5** | 1.13e-4 ± 6.5e-5 |
| Trap | LM-TR | 9.02e-4 ± 6.1e-4 | 1.28e-3 ± 2.4e-3 | **8.76e-5 ± 3.5e-5** | 7.99e-4 ± 5.2e-4 |
| Trap | LM | 1.29e-3 ± 5.9e-4 | 5.53e-5 ± 1.7e-5 | **4.03e-5 ± 5.7e-6** | 2.61e-4 ± 7.2e-5 |
| Trap | Adam | 1.91e-3 ± 7.1e-4 | 2.98e-5 ± 1.0e-5 | **8.56e-5 ± 6.4e-5** | 1.58e-4 ± 5.3e-5 |

- `fcm` **6 konfigürasyonun 6'sında** random'dan daha düşük ortalama ve daha düşük std veriyor. Trapezoid FN'de iyileşme 10–30 kat.
- Makalenin Gauss LM-TR ortalaması 8.07e-7 ± 5.78e-7. `fcm` ile 1.12e-7 ± 8.4e-8, yani hem ortalama hem varyans makaleden düşük.

## 4. P3 / P4 — Trigonometrik veri ve yorumlanabilirlik (10 eşleştirilmiş çift)

| Optimizer | random | lhs | fcm | fcm_xy |
|---|---|---|---|---|
| Adam | 1.44e-2 ± 6.5e-3 | 9.61e-3 * | 1.75e-2 (p = 0.49) | 1.23e-2 (p = 0.19) |
| LM | 2.33e-2 ± 1.5e-2 | 6.66e-3 * | **6.20e-3 *** | 7.11e-3 * |
| LM-TR | 3.32e-2 ± 3.8e-2 | 6.86e-3 * | 2.25e-2 (p = 0.49) | **8.00e-3 *** |

Yorumlanabilirlik ölçütleri (LM-TR satırları; diğer optimizer'lar `p_summary.md` içinde):

| Ölçüt | random | lhs | fcm | fcm_xy |
|---|---|---|---|---|
| Ölü kural, başta | 0.40 | 0 | 0 | 0 |
| Ölü kural, eğitim sonunda | 1.00 | 0 | 0 | 0.10 |
| Hiçbir kuralın ateşlenmediği val. örneği | %4 | %0 | **%6** | %1 |
| Ortalama kapsama oranı ρ (1 = global kural) | 0.51 | **0.70** | 0.50 | 0.59 |
| x₁'i (gürültü) tamamen yok sayan kural (7 kuraldan) | 3.0 | 5.3 | 2.6 | 4.9 |

**Yorum:**
- **H3 (ölü kural) doğrulandı.** Rastgele başlatmada eğitim sonunda ortalama 1 ölü kural kalıyor (7 kuralın %14'ü boşa gidiyor). Önerilen başlatmalarda bu sayı 0–0.1.
- LM ile üç yöntem de anlamlı biçimde daha iyi.
- **LM-TR + fcm'de iyileşme anlamlı değil.** Bunun nedeni tabloda görülüyor: fcm başlatmasında doğrulama örneklerinin **%6'sını hiçbir kural kapsamıyor**. Bu, random'dan bile fazla. FCM'nin bulanık standart sapmasından gelen genişlikler 3 boyutta bazı kümelerde dar kalıyor. Kapsanmayan örnekte çıktı 0'a düşüp büyük hata üretiyor ve bu örneklerde gradyan da sıfır, yani düzeltilemiyor. Bu, katkının çıkış noktası olan "kapsanmayan bölge" sorununun önerilen yöntemin kendisinde de ortaya çıkabildiğini gösteriyor (bkz. §6).
- `fcm_xy`, y'nin değiştiği bölgeleri hedeflediği için trig'de en iyi önerilen varyant; ama sinc'te uçlardaki düşük genlikli dalgaları kaçırıyor (`p2_sinc_preds.png`: düz bölgeler).
- **Yorumlanabilirlik dengesi:** `lhs` her durumda tutarlı, ama kuralları çok genişletiyor (ρ = 0.70, daha global kurallar). Bu, kuralların yerel dilsel anlamını zayıflatıyor. `fcm` ise ρ'yu random seviyesinde (0.50) tutarak yerel kurallar üretiyor.

## 5. P5 — Hesaplama maliyeti

| | Başlatma süresi | Eğitim süresi (LM-TR) |
|---|---|---|
| sinc (16.000 eğitim örneği, 20 kural) | fcm ≈ 2.1 s, lhs ≈ 0.003 s | ≈ 5.5 s |
| trig (6.400 örnek, 7 kural) | fcm ≈ 0.43 s, lhs ≈ 0.003 s | ≈ 8 s |

- FCM başlatması toplam süreyi %19–27 artırıyor. **H5 (ihmal edilebilir maliyet, < %5) doğrulanmadı**; tahmin yanlıştı.
- FCM en fazla 4000 örneklik bir alt kümede çalışıyor. Bu sayı düşürülerek maliyet kolayca azaltılabilir.
- `lhs`'in maliyeti gerçekten ihmal edilebilir.
- Yakınsama 9/20'den 20/20'ye çıktığı için **başarılı bir model elde etmenin beklenen maliyeti** önerilen yöntemle yaklaşık 2 kat daha düşük: random ile ortalama 20/9 ≈ 2.2 deneme gerekiyor, fcm ile 1.

## 6. Tartışma

1. **Ana sonuç:** Başlatmanın giriş uzayını kapsaması, FRBN'nin gradyan tabanlı eğitimini güvenilir hale getiriyor. Etki LM ailesinde en büyük; çünkü LM yerel bir yöntem ve başlangıç noktasına bağımlı.
2. **Asıl etkenin kümeleme değil, kapsama olduğu görülüyor.** Kümeleme yapmayan `lhs` kontrolü birçok konfigürasyonda `fcm` kadar, hatta ondan iyi. Bu bir ablation bulgusu: Kazancın çoğu "her bölgede en az bir kural olması"ndan geliyor; FCM'nin veri yapısını öğrenmesi ek bir avantaj sağlamıyor. Bu yüzden katkı en doğru şekilde "veri güdümlü, kapsama-bilinçli başlatma" olarak tanımlanabilir. `lhs` ve `fcm` bunun iki uygulaması.
3. **FCM'nin zayıf noktası:** Çok boyutlu veride fuzzy std'den gelen genişlikler kapsamayı garanti etmiyor (trig'de %6 kapsanmayan örnek). Olası düzeltmeler: alt genişlik sınırını tam hücre boyu yapmak ya da kapsanmayan örneklere yeni kural eklemek. Bu düzeltmeler bu deneyler görüldükten sonra yapılırsa sonuçlara göre ayar yapmış (p-hacking) oluruz; bu yüzden uygulanmadı, gelecek çalışma olarak bırakıldı.
4. **Hipotezlerin durumu:**

| | Hipotez | Sonuç |
|---|---|---|
| H1 | Yakınsama oranı artar | ✅ (LM-TR 9→20, LM 13→19, Adam 2→10; p < 0.05) |
| H2 | CV'de ortalama ve std düşer | ✅ sinc'te 6/6 konfigürasyon (betimleyici) |
| H3 | Ölü kural 0 olur; trig LM-TR makale seviyesine iner | ✅ ölü kural; ⚠️ LM-TR'de yalnızca `fcm_xy` ve `lhs` ile anlamlı |
| H4 | Daha az ölü kural, daha iyi yorumlanabilirlik | ✅ ölü kural; ⚠️ `lhs` kuralları globalleştiriyor |
| H5 | Maliyet < %5 | ❌ fcm ile %19–27 (lhs ile ihmal edilebilir) |

## 7. Sınırlılıklar

- Deneyler 1 ve 3 boyutlu sentetik verilerle sınırlı. Yüksek boyutlu gerçek veri (ör. makaledeki MEMS) test edilmedi.
- Yalnızca tek katmanlı FN'ler denendi. Çok katmanlı ağlarda yalnızca ilk katman veriden başlatılabiliyor.
- P1 tek bir veri bölmesinde (fold 0) yapıldı. P2'de 5 çift olduğu için istatistiksel güç düşük.
- Süre ölçümleri tek bir dizüstü bilgisayarda CPU üzerinde yapıldı; arka planda başka işlemler çalışmış olabilir.
- y normalizasyonu makalede belirsiz olduğundan trig sonuçlarının makaleyle mutlak karşılaştırması temkinli yorumlanmalı.

## 8. Tekrar üretme

```bash
python -m experiments.proposed.p1_convergence   # ~30 dk
python -m experiments.proposed.p2_sinc_cv       # ~30 dk
python -m experiments.proposed.p3_trig_cv       # ~20 dk
python -m experiments.proposed.p_analysis       # tablolar + şekiller
```
