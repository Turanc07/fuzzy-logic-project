# Checkpoint 4 — Katkı Önerisi: Kapsama-Bilinçli (FCM Tabanlı) Kural Başlatma

**Öğrenci:** Turan Çaldıran (1306220018) · **Makale No:** 44 · Önceki adım: [Checkpoint 3](checkpoint3_baseline_sonuclari.md)

Yönerge §6.1'deki katkı türleri: *parametre/başlatma değişikliği*, *yeni deney senaryosu*, *robustness analizi*.

---

## 1. Ne değiştiriliyor?

FRBN'nin **ilk bulanık katmanının başlatılması.**

| | Baseline (makale ve resmi kod) | Önerilen |
|---|---|---|
| Öncül MF'ler | Kırılma noktaları ~ U(-1, 2) (trapezoid), (μ, σ) ~ U(-1, 1) (Gauss); veriden bağımsız | Kural merkezleri ve genişlikleri **eğitim verisinden** belirleniyor |
| Sonuç MF'leri | Rastgele | Her kuralın kapsadığı örneklerin ateşleme-ağırlıklı y ortalaması ve yayılımı |
| Mimari, parametre sayısı, çıkarım, durulaştırma, optimizer | — | **Aynı** (yalnızca başlangıç noktası değişiyor) |

Karşılaştırılacak başlatma yöntemleri ([src/initialization.py](../src/initialization.py)):

| Kod | Yöntem | Rolü |
|---|---|---|
| `random` | Makaledeki rastgele başlatma | **Baseline** |
| `fcm` | Giriş uzayında **Fuzzy C-Means**: K küme = K kural, genişlik = kümenin bulanık standart sapması | **Ana önerilen yöntem** |
| `fcm_xy` | **Ürün uzayında** ([x, y]) FCM, merkezler girişe izdüşürülüyor; klasik bulanık model tanımlama yaklaşımı | Önerilen yöntemin varyantı |
| `lhs` | Her boyutta kantil ızgarası (Latin hypercube); kümeleme yok | **Kontrol / ablation**: "Sadece kapsama yeterli mi, yoksa veri yapısını öğrenmek mi gerekiyor?" |

Tasarım ayrıntısı: Min t-norm'da bir örneğin bir kuralı ateşlemesi için **her** boyutta o kuralın desteğinde olması gerekiyor. K kural, M boyutta boyut başına K^(1/M) ızgara hücresine karşılık geliyor; bu yüzden en küçük genişlik bu hücre boyuna göre ayarlanıyor. Bu ayar yapılmadan 3 boyutlu trigonometrik veride örneklerin %38'i kapsamsız kalıyordu.

## 2. Neden? (Bilimsel / teknik gerekçe)

1. **Gözlenen problem (Checkpoint 3):**
   - sinc'te FN + LM-TR 20 rastgele başlatmanın yalnızca **9'unda** yakınsıyor; aynı bütçedeki NN 20/20.
   - Başarısız çalıştırmalar ~10⁻³'te takılıyor ve yan dalgalanmaları hiç öğrenemiyor.
   - Trigonometrik veride rastgele başlatma ortalama **1.2 ölü kural** (hiç ateşlenmeyen) bırakıyor.
2. **Mekanizma:**
   - Bir örnekte hiçbir kural ateşlenmezse (w_i = 0) o örnek için tüm öncül parametrelerinin gradyanı sıfır olur.
   - Min t-norm gradyanı yalnızca en küçük üyeliğe aktarır; trapezoidin düz bölgelerinde gradyan da sıfırdır.
   - Sonuç: Kapsanmayan bölge, gradyan tabanlı optimizasyonla (Adam ya da LM) **kurtarılamaz**.
3. **Literatür:** Kural tabanını kümelemeyle (FCM, subtractive clustering) başlatmak bulanık model tanımlamada yerleşik bir yöntem; makale de ilgili çalışmalarda FCM'ye değiniyor. Ancak FRBN makalesi başlatma duyarlılığını hiç raporlamıyor ve yalnızca MEMS'te elle kurulmuş (Wang–Mendel) bir başlatma deniyor. Önerimiz bunun **otomatik ve her veri setine uygulanabilir** karşılığı. Literatür kaynakları rapor aşamasında DOI ile doğrulanarak eklenecek.
4. **Yorumlanabilirliğe olası katkı:** Kurallar verinin bulunduğu bölgelerde başlıyor; ölü ve anlamsız kurallar azalıyor.

## 3. Ön kanıt: Pilot deneme

Betik: `experiments/proposed/p0_pilot.py` · Çıktı: `results/tables/p0_pilot_log.txt`. Fold 0, 100 epoch, LM-TR.

**sinc, Gauss `[1,1,20]`, 8 tohum**

| Başlatma | Başlangıç MSE (medyan) | Son val. MSE (medyan) | En iyi | Yakınsayan (< 1e-5) |
|---|---|---|---|---|
| random | 5.9e-2 | 1.98e-3 | 1.8e-8 | **3/8** |
| lhs | 3.2e-2 | **2.24e-8** | 1.5e-8 | **8/8** |
| fcm | 1.6e-2 | 9.33e-8 | 1.5e-8 | **8/8** |
| fcm_xy | 1.1e-2 | 2.28e-7 | 2.9e-8 | 6/8 |

**trigonometrik, trapezoid `[3,1,7]`, 4 tohum**

| Başlatma | Ölü kural (başlangıçta, ort.) | Son val. MSE (medyan) | En iyi |
|---|---|---|---|
| random | 1.2 | 3.62e-2 | 2.7e-2 |
| lhs | 0 | 7.05e-3 | 1.8e-3 |
| fcm | 0 | 6.78e-3 | **7.3e-4** |
| fcm_xy | 0 | **1.56e-3** | 1.0e-3 |

(Makale, trig LM-TR: ortalama 1.09e-2, en iyi fold 1.26e-3.)

**Ön yorum:**
- Kapsama-bilinçli başlatma sinc'te yakınsama oranını 3/8'den 8/8'e çıkarıyor.
- Trigonometrik veride medyan hatayı 5–23 kat düşürüyor ve makalenin raporladığı seviyeye, hatta altına indiriyor.
- Pilot küçük (4–8 tohum, tek fold). Bu yüzden kesin sonuç değil; asıl değerlendirme aşağıdaki deney tasarımıyla Checkpoint 5'te yapılacak.
- **Dikkat:** 1 boyutlu sinc'te `lhs` deterministik bir eşit ızgaraya dönüşüyor (permütasyon etkisiz). Bu yüzden 8 tohumun hepsi aynı başlangıçtan çıkıyor. Bu bir avantaj (tekrar üretilebilirlik), ama varyans karşılaştırmasında ayrıca belirtilmeli.

## 4. Başarı nasıl ölçülecek? (Deney tasarımı — Checkpoint 5)

**Adil karşılaştırma kuralları:**
- Mimari, parametre sayısı, optimizer, lr, epoch, batch, veri bölmeleri ve tohumlar her yöntemde aynı.
- Başlatma **yalnızca eğitim fold'undan** hesaplanıyor; doğrulama verisi sızmıyor.
- Başlatma süresi de raporlanıyor (ek maliyet).

| Kod | Deney | Konfigürasyonlar | Hipotez |
|---|---|---|---|
| **P1** | Yakınsama oranı (E1b protokolü) | sinc, Gauss `[1,1,20]`, {LM-TR, LM, Adam} × 4 başlatma × 20 tohum, fold 0 | H1: Yakınsama oranı random'a göre anlamlı artar (9/20 → ≥ 18/20) |
| **P2** | sinc 5-katlı CV | {Gauss, Trap} × {Adam, LM, LM-TR} × 4 başlatma, *varied* protokol | H2: Ortalama ve std val. MSE düşer; Adam'a etkisi küçük olabilir |
| **P3** | Trigonometrik 5-katlı CV | Trap `[3,1,7]` × {Adam, LM, LM-TR} × 4 başlatma | H3: Ölü kural sayısı 0 olur; LM-TR makaledeki seviyeye (~1e-2) ya da altına iner |
| **P4** | Yorumlanabilirlik | P3 modellerinde ölü kural, kapsanmayan örnek oranı, MF örtüşmesi, coverage ratio ρ, x₁'i yok sayan kural sayısı | H4: Daha az ölü kural ve daha az aşırı örtüşme |
| **P5** | Maliyet | Başlatma süresi + eğitim süresi | Başlatma maliyeti eğitime göre ihmal edilebilir (< %5) |

**Ölçütler:** Val. MSE (ortalama ± std, medyan, en iyi), yakınsama oranı, ölü kural sayısı, kapsanmayan örnek oranı, süre.

**İstatistik:** Aynı fold/tohum çiftleri üzerinde random'a karşı her yöntem için **Wilcoxon işaretli sıra testi** (eşleştirilmiş, α = 0.05). Yakınsama oranları için Fisher kesin testi.

**Başarısızlık durumu:** Önerilen yöntem bazı konfigürasyonlarda (ör. Adam ya da trapezoid) iyileşme sağlamazsa bu raporlanacak ve nedenleri tartışılacak (yönerge §6.2). Ör. veriye sıkı oturan başlatmanın optimizer'ı yerel bir minimuma kilitlemesi.

## 5. Sınırlılıklar ve kapsam

- Yalnızca **ilk katman** veriden başlatılıyor. Çok katmanlı ağlarda gizli katmanların hedefi bilinmediği için bu katmanlar rastgele kalıyor; kapsam tek katmanlı deneylerle sınırlı.
- FCM'nin kendisi de rastgele başlatmalı bir algoritma; tohum bağımlılığı tamamen ortadan kalkmıyor, azalıyor.
- Yüksek boyutlu verilerde (ör. MEMS, 50 giriş) K^(1/M) genişlik ayarı çok geniş kurallar üretebilir; bu proje kapsamında test edilmiyor.
