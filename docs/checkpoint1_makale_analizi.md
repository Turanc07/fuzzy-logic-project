# Checkpoint 1 — Makale ve Problem Analizi

**Öğrenci:** Turan Çaldıran (1306220018) · **Makale No:** 44
**Referans makale:** H.-I. Lukács, Á. Smidla, T. Fischl, J. Botzheim, "Fuzzy rule-based network: a fuzzy logic-based interpretable and modular machine learning model," *Scientific Reports*, vol. 16, Art. no. 34429, 2026, doi: 10.1038/s41598-026-63687-6.
**Resmi kod:** https://github.com/hunorlukacs/Fuzzy-Network (PyTorch)

> Bu belge makalenin yayıncı sayfasındaki metin ve tablolarından çıkarılmıştır. Sayısal değerler makaledeki tablolardan aynen alınmıştır; sunum/rapora geçirmeden önce PDF ile tekrar karşılaştır.

---

## 1. Problem ve motivasyon

- **Problem:** Derin sinir ağları (NN) yüksek doğruluk sağlar ama "kara kutu"dur; karar mantığı okunamaz. Klasik bulanık kural tabanlı sistemler (FRBS) ve ANFIS yorumlanabilir ama **derin/katmanlı** yapılar kurmak ve ölçeklemek zordur.
- **Amaç:** NN'lerin katmanlı yapısını bulanık kuralların şeffaflığıyla birleştiren, "gri kutu" bir model önermek: **Fuzzy Rule-Based Network (FRBN / FN)**.
- **Uygulama alanı:** Fonksiyon yaklaşımı / regresyon; gerçek dünya örneği olarak MEMS (jiroskop) sensör üretiminde wafer seviyesi test verisinden kalibrasyon parametrelerinin tahmini.
- **Neden bulanık mantık:** Her katmandaki bilgi "IF–THEN" kuralları olarak saklanır → (i) kurallar dilsel olarak okunabilir, (ii) alan bilgisi kural olarak modele başlangıçta verilebilir, (iii) kurallar yerel olduğu için model modülerdir (alt modeller çıkarılabilir).

## 2. Önerilen yöntem (FN)

### 2.1 Mimari
- Ağ, art arda bağlı **bulanık katmanlardan** oluşur: **y = f_L(… f_2(f_1(x)))** (Denk. 10).
- Bir katman `[M, N, K]` ile tanımlanır: M giriş, N çıkış, K kural. Gösterim: `M–N–P–…–R`.
- Katmanda **K×M öncül (antecedent)** matrisi ve **K×N sonuç (consequent)** matrisi vardır. Her çıkış boyutu için öncülleri paylaşan ayrı bir FRBS kurulur → katman doğal olarak **MIMO**'dur.
- Üyelik fonksiyonları zaten doğrusal olmadığından ReLU gibi ek aktivasyon fonksiyonu **kullanılmaz**.

### 2.2 Kural yapısı (Denk. 1)
`R_i: IF x_1 is A_i1 AND … AND x_M is A_iM THEN y_1 is C_i1 AND … AND y_N is C_iN`

### 2.3 Üyelik fonksiyonları
| Tip | Parametre (MF başına) | Çekirdek / Destek |
|---|---|---|
| Trapezoid | a ≤ b ≤ c ≤ d (4) | çekirdek [b,c], destek (a,d) |
| Gauss | μ, σ (2), σ ≥ ε | çekirdek {μ}, pratik destek μ±3σ |

### 2.4 Çıkarım ve durulaştırma
- **Bulanıklaştırma:** her giriş için λ_ij(x_j) (trapezoid: Denk. 2–3, Gauss: Denk. 7).
- **Kural aktivasyonu (t-norm = min):** w_i = min_j λ_ij(x_j) (Denk. 4 / 8).
- **Trapezoid sonuç → Mamdani + Center of Sums (CoS)**, kesilmiş trapezoidlerin alan/moment toplamından kapalı formül (Denk. 5–6).
- **Gauss sonuç → ağırlıklı merkez ortalaması:** y_n = Σ w_i μ̂_in / (Σ w_i + ε) (Denk. 9).

### 2.5 Yorumlanabilirlik metrikleri (post-hoc)
- **Significance score** σ_in(x) (Denk. 11–13): Bir kuralın o girdi için çıktıya göreli katkısı. Trapezoidde kesilmiş sonuç alanı, Gauss'ta normalize ateşleme gücü.
- **Minimum antecedent coverage ratio** ρ_i (Denk. 14): Kural desteğinin evrensel kümeyi ne kadar kapladığı. ~1 ise global kural, ~0 ise lokal kural.

### 2.6 Eğitim / optimizasyon
Eğitilen parametreler MF kırılma noktaları (trapezoid) ya da μ, σ (Gauss).
- **Adam** (1. derece gradyan)
- **Levenberg–Marquardt (LM)**: standart sönümleme ve **trust-region** sönümleme (Denk. 15–19; γ, r_k<0.25 ise ×4, r_k>0.75 ise ÷2; durdurma ‖Jᵀe‖ < 10⁻⁶)
- **BEA** (Bakteriyel Evrimsel Algoritma; gradyansız): bakteriyel mutasyon + gen transferi
- **BMA** (Bacterial Memetic Algorithm) = BEA + yerel gradyan arama (LM/Adam)
- Aynı optimizer'lar karşılaştırma için klasik NN'e de uygulanmıştır.

## 3. Sistem girdileri / çıktıları (deney bazında)

| Deney | Girdi | Çıktı | Not |
|---|---|---|---|
| sinc | x (1-B) | sinc(x) (1-B) | 20.000 nokta |
| Trigonometrik | x₁ (gürültü), x₂, x₃ ∈ [0,1] | y = sin(2πx₂)·cos(2πx₃) | 8000 eğitim örneği |
| MEMS | 50 özellik (konum, reticle, gradyanlar, test istasyonu…) | 18 çıktı (trim parametreleri, max_dist…) | ~800.000 kayıt, anonim |
| MEMS alt küme | Inp F6, Inp F28 | Out F6 | 4 kurallı tek katman, alan bilgisiyle başlatma |

## 4. Deneysel yapı ve temel sonuçlar

### 4.1 Deney 1 — sinc yaklaşımı (FN ve NN)
- FN: 1 katman, 20 kural · NN: 1 gizli katman, 20 nöron · 100 epoch · 5-katlı CV.
- BMA hiperparametreleri: N_gen=1, N_ind=5, N_clones=3, N_inf=2, N_gbm_iter=5, l_gt=4, γ_init=1.

| Model | Konfigürasyon | Val. MSE (ort. ± std) | En iyi fold | Süre [s] | Param. |
|---|---|---|---|---|---|
| **FN** | GradOnly · Gauss · LM-TR | 8.07e-7 ± 5.78e-7 | **6.20e-8** | 20.60 | 72 |
| FN | GradOnly · Gauss · Adam | 3.35e-4 ± 2.60e-4 | 2.05e-5 | 16.63 | 72 |
| FN | GradOnly · Trap · Adam | 2.24e-3 ± 8.34e-4 | 1.46e-3 | 22.06 | 144 |
| **NN** | BMA · tanh · LM-TR | **2.33e-7 ± 1.05e-7** | 1.07e-7 | 58.97 | 61 |
| NN | GradOnly · tanh · LM-TR | 2.82e-7 ± 3.82e-8 | 2.39e-7 | 14.07 | 61 |

**Çıkarım:** En iyi tek fold'u FN alıyor, ama NN eğitimi fold'lar arasında daha kararlı. FN'de BMA genelde **kötüleştiriyor**, NN'de ise yardımcı oluyor.

### 4.2 Deney 2 — Trigonometrik veri
- Mimari arama: 1–3 katman, gizli boyut 1–4, kural 1–7 (100 konfigürasyon; Adam, ≤1000 epoch, early stopping p=10).

| # | Mimari | Train loss | Val loss | Param. |
|---|---|---|---|---|
| 1 | [3,1,7] | 0.005569 | **0.005804** | 112 |
| 2 | [3,1,6] | 0.011311 | 0.011425 | 96 |
| 3 | [3,1,7]-[1,1,7]-[1,1,4] | 0.011315 | 0.011500 | 200 |

- Optimizer karşılaştırması ([3,1,7], 100 epoch, 5-katlı CV): en iyi **GradOnly LM-TR**, val. 0.010896 ± 0.018066, en iyi fold 0.001259. Adam: 0.053763 ± 0.012981. BMA-Adam/LM ise kararsız.
- **Yorumlanabilirlik:** Gürültü özelliği x₁'in MF'leri tüm kurallarda [0,1] üzerinde 1 oluyor; yani model x₁'i kendiliğinden yok sayıyor. Kurallar dilsel olarak okunabiliyor (ör. R₁: x₂ Low ve x₃ High → y High).
- **Ablation (korelasyonlu gürültü x₁ = 0.3x₂ + ε, Gauss, 10 kural):**

| ε (std) | 3 özellik MSE | 2 özellik MSE | ΔMSE |
|---|---|---|---|
| 0.1 | 0.000607 | 0.001167 | −0.000561 |
| 0.5 | 0.000763 | 0.001079 | −0.000316 |
| 1.0 | 0.000987 | 0.001100 | −0.000113 |
| 2.0 | 0.001291 | 0.001354 | −0.000062 |

### 4.3 Deney 3 — MEMS
- Gauss FN, 100 epoch, 5-katlı CV: en iyi **GradOnly Gauss LM-TR / LM**, val. MSE 1.50e-4 ± 4.0e-6 (24 parametre).
- Alan bilgisiyle (Wang–Mendel) başlatma, rastgele başlatmaya göre daha düşük başlangıç hatası ve daha kararlı yakınsama sağlıyor (0.000153 ± 0.000005). Rastgele başlatmalı Adam'da 5 fold'dan 2'si yakınsamadı.
- Significance ile budama: Bir alt bölgede baskın kural tek başına bırakıldığında ΔMSE = 0; baskın kural atıldığında hata belirgin artıyor. SHAP ile tutarlı açıklamalar elde ediliyor.

## 5. Eleştirel değerlendirme

**Güçlü yönler**
- Açık matematiksel tanım (kapalı formül CoS), resmi kod mevcut.
- Yorumlanabilirlik nicel metriklerle ölçülüyor (significance, coverage) ve SHAP ile karşılaştırılıyor.
- Alan bilgisiyle başlatma ve modüler alt model çıkarma pratik açıdan değerli.

**Sınırlılıklar / zayıf yönler**
- Deneylerin çoğu **sentetik ve düşük boyutlu** (1-B sinc, 3-B trig). Standart açık benchmark (UCI vb.) yok; MEMS verisi anonim ve tekrar üretimi zor.
- Karşılaştırma yalnız tek gizli katmanlı NN ile yapılmış. **ANFIS, TSK, karar ağacı, GBM** gibi yorumlanabilir ya da güçlü baseline'lar yok.
- Varyans yüksek: FN-LM-TR trig'de 0.0109 ± 0.0181 (std > ortalama). Sonuç büyük ölçüde başlatmaya bağlı.
- **min t-norm** gradyanı yalnızca tek bir öncüle akıtır (seyrek gradyan), trapezoidin düz bölgelerinde gradyan sıfırdır. Bu, trapezoid + Adam sonuçlarının neden zayıf olduğunu açıklayabilir (bizim hipotezimiz, makalede tartışılmıyor).
- Eğitim sonrası MF'ler çok örtüşebiliyor, bu da yorumlanabilirliği azaltıyor (yazarlar Ruspini bölüntüsünü gelecek çalışma olarak öneriyor).

**Makalede fark ettiğim tutarsızlıklar (sunumda/raporda dikkatle ele alınmalı)**
1. Tablo 1(a)'nın başlığı "NN configurations" diyor ama içerik **FN** sonuçları (Gauss/Trap MF).
2. sinc için 20 kurallı FN'nin parametre sayısı Gauss'ta 72, trapezoidde 144 olarak verilmiş. Beklenen değerler 20×(2+2)=80 ve 20×(4+4)=160. Sayılar 18 kurala uyuyor; kural budama ya da sayım farkı olabilir, kodda kontrol edilmeli.
3. MEMS metninde "BMA modelleri ~10⁻³ MSE ile GradOnly'den bir mertebe daha iyi" deniyor; ama Tablo 5'te en iyi sonuçlar **GradOnly** LM-TR/LM (1.5e-4) ve sonuç bölümü de LM-TR'yi en iyi ilan ediyor. Metin ile tablo çelişiyor.
4. Trust-region oranında (Denk. 17) hata normları yerine parametre normu ‖b‖² yazılmış; muhtemelen gösterim hatası.
5. Sonuç bölümünde trigonometrik veri "triangular dataset" olarak anılıyor.

## 6. Proje kapsamı önerisi (MRR taslağı — hocayla teyit edilecek)

**Yeniden gerçekleştirme (baseline)**
1. FN katmanını kendim yazmak (PyTorch): trapezoid ve Gauss MF, min t-norm, CoS ve ağırlıklı ortalama durulaştırma, çok katmanlı yapı.
2. Eğitim: Adam ve LM trust-region (BEA/BMA isteğe bağlı, zaman kalırsa).
3. **Deney 1 (sinc)** ve **Deney 2 (trig, [3,1,7])** tekrarı, 5-katlı CV, Tablo 1 ve Tablo 3 ile karşılaştırma.
4. Significance score ve coverage ratio hesabı, kural görselleştirmesi.
5. MEMS deneyi kapsam dışı (veri boyutu/erişim); gerekçesi raporda belirtilecek.

**Özgün katkı adayları (Checkpoint 4'te biri seçilecek)**
- **A. t-norm karşılaştırması:** min ve çarpım (product). Gerekçe: çarpım tüm öncüllere gradyan akıtır, gradyan tabanlı eğitimde kararlılık ve varyansın iyileşmesi beklenir.
- **B. Ruspini kısıtlı MF bölüntüsü:** Yazarların önerdiği gelecek çalışma. Doğruluk ile yorumlanabilirlik (MF örtüşmesi) dengesini ölçmek.
- **C. Sugeno (TSK) sonuç katmanı ile Mamdani karşılaştırması.**
- **D. Gerçek ve açık bir benchmark (ör. UCI regresyon verisi) üzerinde FN, ANFIS ve NN karşılaştırması.**
- **E. Gürültü/robustness analizi** (girdiye Gauss gürültüsü, aykırı değerler).

**Ölçütler:** Val. MSE / RMSE (ort. ± std, en iyi fold), eğitim süresi, parametre sayısı, yakınsamayan fold sayısı; yorumlanabilirlik için ortalama MF örtüşmesi ve aktif kural sayısı.
