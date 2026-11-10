# Checkpoint 2 — Uygulama Planı

**Öğrenci:** Turan Çaldıran (1306220018) · **Makale No:** 44 · **Önceki adım:** [Checkpoint 1](checkpoint1_makale_analizi.md)

Bu belge yeniden gerçekleştirmenin **nasıl** yapılacağını tanımlar: geliştirme ortamı, veri setleri, üyelik fonksiyonları, kural tabanı, eğitim, deney planı ve başarı ölçütleri.

---

## 1. Geliştirme ortamı

| Bileşen | Seçim | Gerekçe |
|---|---|---|
| Dil | Python 3.14.5 | Makalenin resmi kodu Python/PyTorch |
| Çatı | PyTorch 2.15.0 (CPU) | Otomatik türev, LM için Jacobian (`torch.func`) |
| Diğer | numpy 2.5.4, pandas 3.1.1, matplotlib 3.11.2, scikit-learn 1.9.1, pytest 9.2.0 | Veri, tablo, grafik, test |
| Donanım | Windows 11, CPU (RTX 3060 mevcut ama gerekli değil) | Modeller küçük (≤ 200 parametre). CPU'da sonuçlar deterministik ve her makinede tekrar üretilebilir |
| Tekrar üretilebilirlik | Sabit tohumlar (`set_seed`), sabitlenmiş sürümler ([requirements.txt](../requirements.txt)), `pytest` testleri | Yönerge §11 |

Kurulum:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 2. Kod kaynağı politikası

- FN katmanı, çıkarım, durulaştırma, LM trust-region, metrikler ve deney betiklerini **sıfırdan yazıyorum**.
- Resmi depo ([hunorlukacs/Fuzzy-Network](https://github.com/hunorlukacs/Fuzzy-Network)) yalnızca **referans** olarak okunuyor; makalede yazmayan ayrıntılar (veri aralığı, başlatma aralıkları) oradan alınıyor ve aşağıda belgeleniyor.
- Resmi depodan doğrudan kod alınırsa ilgili dosyanın başında kaynak gösterilecek (yönerge §16).

## 3. Veri setleri

| Ad | Tanım | Boyut | Kaynak / not |
|---|---|---|---|
| **sinc** | x ∈ [-1, 1] eşit aralıklı, y = sinc(10x) = sin(10πx)/(10πx) | 20.000 × (1→1) | Makale: 20.000 nokta. Frekans ve aralık resmi notebook'tan |
| **trig** | x₂, x₃ ~ U[0,1]; x₁ ~ U[0,1] (saf gürültü); y = (sin(2πx₂)·cos(2πx₃) + 1)/2 | 8000 × (3→1) | Makale Denk. 20. y'nin [0,1]'e normalize edildiği belirtiliyor; dönüşüm (y+1)/2 olarak varsayıldı |
| **trig-corr** | x₁ = 0.3·x₂ + ε, ε ~ N(0, s²), s ∈ {0.1, 0.5, 1.0, 2.0}; x₁ min-max ölçeklenir | 8000 × (3→1) | Ablation, makale Tablo 4 |
| MEMS | — | — | **Kapsam dışı:** 800 bin kayıt, anonim endüstriyel veri |

- Bölme: **5-katlı çapraz doğrulama**, karıştırılmış, sabit tohum (`kfold_indices`).
- Uygulama: [src/data.py](../src/data.py) (hazır, testleri geçiyor: [tests/test_data.py](../tests/test_data.py)).

## 4. Bulanık sistem tasarımı

### 4.1 Üyelik fonksiyonları
| Tip | Parametre | Kısıt | Başlatma (resmi koddaki gibi) |
|---|---|---|---|
| Trapezoid | (a, b, c, d) | Her ileri geçişte sıralanır → a ≤ b ≤ c ≤ d | U(-1, 2) |
| Gauss | (μ, σ) | σ ← max(\|σ\|, ε) | U(-1, 1) |

Not: Makale ve kod, Gauss sonuç katmanında yalnızca μ̂'yü kullanıyor (Denk. 9). σ̂ çıktıyı etkilemiyor, yani gradyanı sıfır. Parametre sayısını raporlarken bunu ayrıca belirteceğim.

### 4.2 Çıkarım zinciri (her katman)
1. Bulanıklaştırma: λ_ij(x_j) (trapezoid Denk. 2–3 / Gauss Denk. 7)
2. Kural ateşleme: w_i = min_j λ_ij (t-norm = min)
3. Durulaştırma:
   - Trapezoid sonuç → **Mamdani + Center of Sums**, kapalı formül (Denk. 5–6)
   - Gauss sonuç → **ağırlıklı merkez ortalaması** (Denk. 9)
4. Katmanlar zincirlenir: y = f_L(…f_1(x)), ek aktivasyon fonksiyonu yok.

### 4.3 Kural tabanları (mimariler)
| Deney | Mimari `[giriş, çıkış, kural]` | MF | Eğitilen parametre sayısı (beklenen) |
|---|---|---|---|
| sinc | `[1,1,20]` | Gauss / Trapezoid | 80 / 160 (makale 72 / 144 diyor, kontrol edilecek) |
| trig | `[3,1,7]` | Trapezoid | 7·(3+1)·4 = **112** (Tablo 2 ile aynı) |
| trig mimari araması (küçültülmüş) | 1–2 katman, 1–7 kural | Trapezoid | — |
| ablation | `[3,1,10]` ve `[2,1,10]` | Gauss | 60 / 40 (makaledeki sayım) |

Kural tabanı **veriden öğrenilir** (rastgele başlatma + optimizasyon). Kurallar eğitimden sonra dilsel etiketlere (Low/Mid/High) çevrilerek yorumlanacak.

### 4.4 Yorumlanabilirlik metrikleri
- **Significance score** σ_in(x) (Denk. 11–13): Trapezoidde kesilmiş sonuç alanı sayısal integralle hesaplanır, Gauss'ta normalize w_i kullanılır.
- **Minimum antecedent coverage ratio** ρ_i (Denk. 14).
- **Alt model çıkarma:** sinc'te [-0.25, 0.25] aralığında en anlamlı k kuralın tutulması (makale Şekil 7).

## 5. Eğitim / optimizasyon

| Optimizer | Uygulama | Ayarlar |
|---|---|---|
| Adam | `torch.optim.Adam` | lr = 0.01 (resmi kod), full-batch |
| LM (standart) | Kendi uygulamam: s = −(JᵀJ + γI)⁻¹Jᵀe | γ: başarılı adımda ÷10, başarısızda ×10 |
| LM trust-region | Kendi uygulamam (Denk. 15–19) | γ_init = 1; r<0.25 → 4γ; r>0.75 → γ/2; ‖Jᵀe‖ < 10⁻⁶ → dur |
| BEA / BMA | **İsteğe bağlı (zaman kalırsa)** | Makaledeki hiperparametreler: N_ind=5, N_clones=3, N_inf=2, l_gt=4 |

- Jacobian: `torch.func.jacrev` + `vmap` ile örnek başına gradyan. Gerekirse veri alt kümesi kullanılır; sinc için 20k × 80 küçük bir matris.
- Epoch: 100 (makale). Kayıp: MSE.
- Trapezoid parametrelerindeki sıralama kısıtı her ileri geçişte uygulanır.

**MRR önerisi:** FN (iki MF tipiyle) + Adam + LM trust-region + 2 deneyin tekrarı + yorumlanabilirlik metrikleri. BEA/BMA kapsamın dışında tutulup isteğe bağlı bırakılabilir; bu durum raporda gerekçelendirilecek.

## 6. Deney planı

| Kod | Deney | Konfigürasyonlar | Karşılaştırılan makale sonucu |
|---|---|---|---|
| **E1** | sinc, FN ve NN | FN: {Gauss, Trap} × {Adam, LM, LM-TR}; NN (1×20 nöron): {tanh, ReLU} × {Adam, LM, LM-TR}; 5-katlı CV | Tablo 1 (ör. FN Gauss LM-TR 8.07e-7 ± 5.78e-7; NN tanh LM-TR 2.82e-7) |
| **E2** | trig, optimizer karşılaştırması | `[3,1,7]` Trap × {Adam, LM, LM-TR}; 5-katlı CV | Tablo 3 (LM-TR en iyi fold 0.001259) |
| **E3** | trig, küçük mimari araması | 1–2 katman × 2–7 kural, Adam + early stopping | Tablo 2 (`[3,1,7]` en iyi) |
| **E4** | Korelasyonlu gürültü ablation | 3 özellik ve 2 özellik, s ∈ {0.1, 0.5, 1, 2}, 3 tekrar | Tablo 4 |
| **E5** | Yorumlanabilirlik | Significance haritası, coverage, x₁ MF'lerinin [0,1]'de 1 olduğunun kontrolü, sinc alt model çıkarma | Şekil 6–10, 12 |

Kayıt formatı: Her deney `results/tables/<deney>.csv` ve `results/figures/<deney>_*.png` üretir. Betikler `experiments/baseline/` altında, tek komutla çalışacak şekilde yazılacak.

## 7. Başarı ölçütleri

- **Doğruluk:** Doğrulama MSE (ortalama ± std ve en iyi fold), eğitim MSE.
- **Verim:** Eğitim süresi [s], parametre sayısı.
- **Kararlılık:** Yakınsamayan fold sayısı (ör. son MSE > 10 × medyan).
- **Tekrar üretim başarısı (benim tanımım):** Makaledeki **sıralamaların** korunması (ör. FN'de LM-TR > Adam; trig'de x₁'in yok sayılması) ve MSE'nin aynı mertebede olması. Birebir sayı beklenmiyor: tohum, başlatma ve donanım farklı.

## 8. Makale ile resmi kod arasında bulunan farklar

Uygulamada hangisinin esas alındığı belirtilmiştir.

| Konu | Makale | Resmi notebook | Benim seçimim |
|---|---|---|---|
| sinc nokta sayısı | 20.000 | 200.000 (batch 20.000) | 20.000 (makale) |
| Çapraz doğrulama | 5-katlı CV | Yayımlanan notebook'ta CV yok, eğitim MSE raporlanıyor | 5-katlı CV (makale) |
| BMA hiperparametreleri (sinc) | N_ind=5, N_clones=3 | n_ind=10, n_clone=5 | Makale |
| LM uygulaması | Algoritma 2 | Harici `torch-levenberg-marquardt` kütüphanesi | Kendi uygulamam (Denk. 15–19) |
| Gauss sonuç σ̂ | Kullanılmıyor (Denk. 9) | Parametre olarak var, kullanılmıyor | Parametre sayımında ayrı belirtilecek |

## 9. Riskler ve önlemler

| Risk | Önlem |
|---|---|
| Trapezoidin düz bölgelerinde gradyan 0 → Adam/LM takılabilir | Birden fazla tohum ve başlatma aralığı; bu durum katkı (Checkpoint 4) için de bir motivasyon |
| LM'de Jacobian bellek/zaman maliyeti | Parametre sayısı küçük; gerekirse alt örnekleme |
| Makale sonuçlarının birebir tutmaması | Sıralama ve mertebe karşılaştırması; farkların nedenleri raporda tartışılacak |
| Zaman | BEA/BMA isteğe bağlı; MRR önce tamamlanacak |

## 10. Planlanan kod yapısı

```
src/
  data.py            # veri üreticileri + k-fold   (hazır)
  membership.py      # trapmf, gaussmf
  fuzzy_layer.py     # FuzzyLayer: min t-norm, CoS / ağırlıklı ortalama
  fuzzy_network.py   # katmanların zincirlenmesi, parametre sayımı, kural yazdırma
  nn_baseline.py     # 1 gizli katmanlı MLP
  optim_lm.py        # LM ve LM trust-region
  train.py           # eğitim döngüsü, CV, kayıt
  interpret.py       # significance, coverage, alt model çıkarma, dilsel etiketler
experiments/baseline/
  e1_sinc.py  e2_trig_optimizers.py  e3_trig_arch.py  e4_ablation.py  e5_interpret.py
tests/
  test_data.py (hazır)  test_membership.py  test_fuzzy_layer.py  test_optim_lm.py
```

## 11. Sonraki adım (Checkpoint 3)

Önce `membership.py`, `fuzzy_layer.py` ve `fuzzy_network.py` yazılacak; elle hesaplanmış küçük örneklerle test edilecek (CoS formülü, sayısal integralle karşılaştırılacak). Ardından optimizer'lar ve E1–E2 deneyleri gelecek.
