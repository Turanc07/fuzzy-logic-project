# Sözlü Savunma ve Canlı Demo Hazırlığı

**Öğrenci:** Turan ÇALDIRAN (1306220018) · **Makale No:** 44

Yönerge §14 savunmada şunların sorulabileceğini söylüyor: kuralların işlevi, üyelik fonksiyonu seçimleri, parametrelerin etkileri, algoritmalar, değişikliklerin gerekçeleri, sonuçların nedenleri, kodun bölümleri ve sınırlılıklar. Aşağıdaki cevaplar repodaki sonuç dosyalarına dayanıyor. **Ezberlemek yerine her birini kodda ve sonuç tablolarında kendin kontrol et.**

---

## 1. Canlı demo akışı (≈ 3 dk)

Sunumdan **önce** (internet ve eğitim süresine bağlı kalmamak için) bir kez çalıştır:
```bash
.venv\Scripts\activate
python -m demo.frbn_demo compare --no-show
python -m demo.frbn_demo train --task trig --init fcm
python -m demo.frbn_demo train --task trig --init random
python -m demo.frbn_demo train --task sinc --init fcm
```
Modeller `demo/models/` altında önbelleğe alınır; `rules` ve `predict` anında çalışır.

Sunumda:
1. `python -m demo.frbn_demo compare` → aynı tohum, random (MSE ≈ 2e-3, sağdaki dalgalar düz) ve fcm (≈ 1e-7). ~20 sn eğitim yapar.
2. `python -m demo.frbn_demo rules --task trig --init fcm` → 7 kural.
3. `python -m demo.frbn_demo predict --task trig --init fcm --x 0.2 0.15 0.5` → adım adım açıkla:
   - "Her kural için girişlerin üyelik dereceleri hesaplanıyor (bulanıklaştırma)."
   - "VE bağlacı minimum: ateşleme gücü w, en küçük üyelik."
   - "Ateşlenen kuralların payları; Center of Sums ile durulaştırılmış çıktı ve gerçek değer."
4. Hoca bir girdi verirse: `--x a b c` (trig, [0,1] aralığında üç sayı) ya da `--task sinc --x a` ([-1, 1]).

Not: Demodaki trig modeli (tohum 0, LM-TR) slayttaki P6 tablosundaki modelden (tohum 100, LM) farklı bir eğitim çalıştırması. Bu yüzden kurallar birebir aynı değil. Sorulursa: "Farklı başlatma ve optimizer'la eğitilmiş başka bir model; kuralların yönleri benzer ama kural numaraları ve sınırları değişiyor."

**Olası demo soruları:**
- *"Hiçbir kural ateşlenmezse ne olur?"* → Çıktı 0'a düşer (Denk. 4'te pay da payda da 0) ve gradyan akmaz. Demo bunu "!!" satırıyla gösterir. Veri aralığı dışında bir girdi dene: `--x 5 5 5`.
- *"Bu kuralı kapatırsam ne olur?"* → Yalnızca o kuralın ateşlendiği bölgede çıktı değişir (yerellik / modülerlik). Makale bunu alt-model çıkarma ile gösteriyor (makale Şekil 6–8).

## 2. Kod haritası ("Bu dosya ne yapıyor?")

| Dosya | Görev | Savunmada söylenecek tek cümle |
|---|---|---|
| `src/membership.py` | `trapmf`, `gaussmf` | Trapezoid parametreleri her adımda sıralanıyor, böylece a ≤ b ≤ c ≤ d hep geçerli |
| `src/fuzzy_layer.py` | `firing_strengths`, `cos_defuzz_trap`, `weighted_avg_defuzz` | Ateşleme = üyeliklerin minimumu; CoS kapalı formülü kesilmiş trapezoid alan ve momentinden |
| `src/fuzzy_network.py` | `FuzzyNetwork`, `rules()` | Katmanlar art arda; arada aktivasyon yok |
| `src/optim_lm.py` | `LevenbergMarquardt.step` | s = −(JᵀJ + γI)⁻¹Jᵀe; hatayı azaltmayan adım reddediliyor; trust-region'da γ, r oranına göre |
| `src/train.py` | `train_model`, `cross_validate` | Her epoch 10 mini-batch, her batch bir optimizer adımı |
| `src/initialization.py` | `initialize`, `fuzzy_c_means`, `dead_rules` | **Katkı:** kurallar FCM ya da kantil ızgarasıyla verinin üzerine; sonuçlar ateşleme-ağırlıklı y ortalaması |
| `src/experiment.py` | `run_one`, `run_experiment` | Aynı tohum/fold ile adil karşılaştırma, tüm sayılar CSV'ye |
| `experiments/proposed/p_analysis.py` | Wilcoxon, Fisher | Random'a karşı eşleştirilmiş testler |

## 3. Olası sorular ve cevap iskeletleri

### Kurallar ve üyelik fonksiyonları
**S: Neden min t-norm? Çarpım olsa ne olurdu?**
Makale min kullanıyor; ben makaleye sadık kaldım. Min'in sonucu: gradyan yalnızca en küçük üyeliğe akıyor ve kural, örnek her boyutta desteğindeyse ateşleniyor. Çarpım tüm öncüllere gradyan akıtır, ama yüksek boyutta ateşlemeler çok küçülür (Cui vd., 2021). Bu karşılaştırmayı yapmadım; gelecek çalışma.

**S: Neden trapezoid Gauss'tan kötü?**
Trapezoidin çekirdeğinde ve destek dışında türev sıfır; gradyan tabanlı eğitimde bu bölgelerde parametreler hareket etmiyor. Gauss her yerde türevlenebilir. Sonuçlarda da trapezoid FN Gauss'tan bir mertebe kötü (E1, makaleyle tutarlı).

**S: Kural sayısını nasıl seçtiniz?**
Makaleyi izledim: sinc'te 20 kural (20 nöronlu ağla adil karşılaştırma), trig'de [3,1,7] (makalenin mimari aramasının sonucu, Tablo 2). Kendi mimari aramamı yapmadım.

**S: "herhangi" etiketi ne demek?**
O girişin üyelik fonksiyonu [0,1]'in tamamında ≈1. Min'in birim elemanı olduğu için kural o girişi yok sayıyor. Trig'de gürültü özelliği x₁ çoğu kuralda "herhangi" çıkıyor; makalenin "x₁'i filtreler" iddiası bu.

### Algoritmalar ve parametreler
**S: LM ile Adam farkı?**
Adam birinci derece, adım boyu öğrenme oranıyla sabit. LM, Jacobian'la ikinci dereceye yaklaşıyor: γ küçükken Gauss–Newton (hızlı), büyükken gradyan inişi (güvenli). Küçük modellerde (≤160 parametre) LM çok daha hızlı yakınsıyor. Bu yüzden NN'de 2e-7, Adam'da 1e-2.

**S: Trust-region ne değiştiriyor?**
γ'yı, gerçekleşen hata azalmasının doğrusal modelin öngördüğüne oranına (r) göre güncelliyor: r < 0.25 → 4γ, r > 0.75 → γ/2. Standart LM'de ise başarılı adımda γ/10, başarısızda 10γ.

**S: Mini-batch boyutu neden eğitim kümesinin 1/10'u?**
Resmi kodda 200k veri ve 20k batch, yani aynı oran. Full-batch 100 epoch = 100 LM adımı yetmedi (ön denemede 2e-2'de kaldı); 1000 adım gerekiyor.

**S: FCM'de m = 2 neden?**
Standart değer. Daha büyük m kümeleri daha bulanık yapar, genişlikler büyür. m'nin etkisini test etmedim.

### Katkının gerekçesi ve sonuçların nedenleri
**S: FRBN neden başlatmaya bu kadar duyarlı?**
İki mekanizma var:
1. Bir kuralın parametreleri yalnızca o kuralın ateşlendiği örneklerden gradyan alıyor. Hiç ateşlenmeyen kural (ölü kural) hiç öğrenemiyor. Trig'de rastgele başlatmada eğitim sonunda ortalama 1 ölü kural var.
2. LM yerel bir yöntem; kuralları başladıkları yerden uzağa taşıyamıyor. sinc'te başarısız modelde 20 merkezden 3'ü veri aralığı dışında, 2'si aynı noktada; 0.1–0.3 aralığında σ ≈ 1.1 olan tek bir geniş kural var, yani dalgalanmayı temsil edecek dar kural yok.

**Dikkat:** İlk yazdığım açıklama "o bölgede hiç kural ateşlenmiyor" idi; modeli ölçünce bunun yanlış olduğunu gördüm ve düzelttim. Sorulursa dürüstçe anlat: "İlk hipotezimi ölçümle test ettim, yanlış çıktı, düzelttim."

**S: Neden NN'de bu sorun yok?**
Bir nöronun aktivasyonu her örnekte sıfırdan farklı (tanh), yani her parametre her örnekten gradyan alıyor; kör bölge yok. Kuralların yerelliği FRBN'yi yorumlanabilir yapan şey, ama aynı zamanda optimizasyonu kırılgan yapıyor.

**S: Bu yeni bir fikir mi?**
Hayır, kümelemeyle başlatma bilinen bir yöntem: Chiu 1994, ürün uzayı kümelemesi, özellikle Shi vd. 2021 (FCM-RDpA, TSK için). Katkım: (1) FRBN'nin başlatma duyarlılığını nicel göstermek (makale raporlamıyor), (2) Mamdani + min t-norm yapısına uyarlamak (K^(1/M) genişlik ayarı, ateşleme-ağırlıklı sonuç başlatma), (3) kazancın kümelemeden mi kapsamadan mı geldiğini lhs kontrolüyle ayırt etmek.

**S: lhs fcm kadar iyiyse neden fcm önerilen yöntem?**
Dürüst cevap: düşük boyutta fark yok, kazancın kaynağı kapsama. fcm'yi ana yöntem seçtim çünkü (a) veri dağılımına uyum sağlıyor (Shi vd. yüksek boyutta ızgaradan iyi olduğunu gösterdi), (b) lhs 1-D'de deterministik bir ızgaraya dönüşüyor, (c) lhs kuralları globalleştiriyor (ρ = 0.70), fcm yerel kurallar üretiyor (ρ ≈ 0.50). Ama bunu sonuçlardan sonra değil, önceden (Checkpoint 4) seçtim.

**S: Trig'de LM-TR + fcm neden anlamlı değil?**
fcm doğrulama örneklerinin %6'sını kapsamıyor (random: %4). FCM'nin bulanık std'sinden gelen genişlikler 3 boyutta bazı kümelerde dar kalıyor. Kapsanmayan örnekte çıktı 0 → büyük hata. Düzeltmedim, çünkü sonuçları gördükten sonra yöntemi ayarlamak sonuçlara göre ayar yapmak (p-hacking) olurdu.

**S: Trig'de makalenin sonuçlarını neden tekrarlayamadınız?**
Üç olası neden: (1) y'nin nasıl normalize edildiği makalede yok; biz (y+1)/2 kullandık, bu MSE'yi ölçekler. (2) Makale harici bir LM kütüphanesi kullanıyor. (3) Başlatma: sabit tohum 42 bizde farklı bir başlangıç üretir. Hangisinin etkili olduğunu ayrıştırmadım.

**S: İstatistiksel testleri neden böyle seçtiniz?**
Aynı tohum ve fold'da eşleştirilmiş ölçümler → eşleştirilmiş Wilcoxon (normal dağılım varsayımı yok, log MSE üzerinde). Yakınsama sayıları 2×2 tablo → Fisher kesin testi. P2'de 5 çiftle Wilcoxon'un en küçük p'si 0.0625, o yüzden P2 betimleyici; anlamlılık P1 (20 çift) ve P3 (10 çift).

### Sınırlılıklar
- 1 ve 3 boyutlu sentetik veri; MEMS ve yüksek boyut test edilmedi.
- BMA kapsam dışı.
- Tek katmanlı ağlar; çok katmanda sadece ilk katman veriden başlatılabiliyor.
- Sonuç kümeleri [0,1] dışına taşıyor; dilsel yorum yalnızca yön olarak.
- FCM'nin maliyeti %19–27 (tahminim %5'ti, yanlış çıktı).

## 4. Rakamlar (hızlı bakış)

| | |
|---|---|
| sinc, makale FN Gauss LM-TR | 8.07e-7 ± 5.78e-7 (en iyi 6.2e-8) |
| sinc, bizim FN Gauss LM-TR + fcm (5 fold) | 1.12e-7 ± 8.4e-8 |
| Yakınsama (20 başlatma, LM-TR) | random 9/20 → fcm 20/20 (Fisher p < 0.001) |
| Medyan hata (LM-TR) | 1.98e-3 → 1.05e-7 |
| Trig LM ortalama | 2.33e-2 → 6.20e-3 (p = 0.014) |
| Trig ölü kural (eğitim sonu) | 1.0 → 0 |
| Test sayısı | 47 |
| Toplam eğitim çalıştırması | 600+ |

## 5. Etik ve yapay zekâ kullanımı
Yönerge §15'e göre kodun ve açıklamaların öğrenci tarafından anlaşılmış ve doğrulanmış olması gerekiyor. Raporda yapay zekâ kullanımı beyan edildi. Savunmadan önce her dosyayı açıp ana fonksiyonları kendin takip et; özellikle `firing_strengths`, `cos_defuzz_trap`, `LevenbergMarquardt.step` ve `initialize`.
