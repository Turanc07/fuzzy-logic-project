# Literatür Taraması

**Öğrenci:** Turan ÇALDIRAN (1306220018) · **Makale No:** 44

Bu belge raporun *Related Work* bölümünün ve sunumdaki literatür slaytının temelidir. Tüm kaynakların künyeleri **Crossref / arXiv / PMLR kayıtlarından doğrulanmıştır** (tam liste: [references/README.md](../references/README.md)). Güncel çalışmaların özetleri yayıncı ya da arXiv özetlerinden alınmıştır; tam metinleri okunmamış olanlar yalnızca özet düzeyinde anılmıştır.

---

## 1. Temel kavramlar

| Kaynak | Katkı | Projedeki rolü |
|---|---|---|
| Zadeh, 1965 | Bulanık küme kavramı | Teorik altyapı |
| Mamdani & Assilian, 1975 | Dilsel kurallarla bulanık denetleyici (Mamdani çıkarımı) | FRBN, Mamdani çıkarımı + CoS durulaştırma kullanır |
| Takagi & Sugeno, 1985 | Sonuç kısmı fonksiyon olan (TSK) bulanık modeller | Literatürdeki eğitim yöntemlerinin çoğu TSK üzerine; FRBN'den farkı |
| Ruspini, 1969 | Üyelik derecelerinin toplamı 1 olan bölüntü | Makalenin önerdiği gelecek çalışma (Ruspini bölüntüsü) |

## 2. Nöro-bulanık ve derin bulanık modeller

| Kaynak | Yöntem | Bulgu / özellik | Referans makaleyle ilişkisi |
|---|---|---|---|
| Jang, 1993 (ANFIS) | Sinir ağı yapısında TSK, hibrit öğrenme | Yorumlanabilir, ama tek katmanlı ve çok girişte kural sayısı patlıyor | FRBN'nin aşmayı hedeflediği temel model |
| Kerr-Wilson & Pedrycz, 2020 | Hiyerarşik (kademeli) TSK kural tabanı | Kural sayısını ve karmaşıklığını azaltıyor, çoğu durumda doğruluğu artırıyor | FRBN gibi katmanlı; ama kademeler arası bilgi akışı farklı |
| Huang vd., 2022 | FCM bölüntülü derin "patch learning" sınıflandırıcıları | Izgara bölüntüsünün yol açtığı kural patlamasını FCM ile önlüyor | Derin bulanık yapıda FCM ile bölüntü fikri |
| Talpur vd., 2023 (sistematik tarama) | 2015–2020 derin nöro-bulanık sistemler | Yorumlanabilirlik ile makul doğruluk arasındaki dengeyi ve araştırma boşluklarını özetliyor | Alanın genel çerçevesi |
| Yeganejou vd., 2023 (DCNFIS) | Derin evrişimli nöro-bulanık çıkarım sistemi | CNN'lerle rekabetçi doğruluk, kurallardan saliency açıklamaları | Görüntü alanında derin bulanık model; FRBN'nin "evrişimli FN" gelecek çalışmasıyla ilişkili |
| Zhang vd., 2026 (FRDRL) | TSK öncül kısmıyla türevlenebilir temsil öğrenme | Kara kutu temsil öğrenmeye yorumlanabilir alternatif | Kural tabanlı modellerin derin öğrenmeyle birleştirilmesine güncel örnek |
| **Lukács vd., 2026 (FRBN)** | **Mamdani kural katmanlarının zincirlenmesi** | sinc'te NN ile rekabetçi; significance ve coverage metrikleriyle modülerlik | **Referans makale** |

## 3. Bulanık sistemlerin eğitimi

| Kaynak | Yöntem | Projedeki rolü |
|---|---|---|
| Levenberg, 1944; Marquardt, 1963 | Sönümlü Gauss–Newton (LM) | `src/optim_lm.py` |
| Kingma & Ba, 2014 | Adam | Birinci derece optimizer |
| Nawa & Furuhashi, 1999 | Bakteriyel evrimsel algoritma (BEA) | Makalenin BMA'sının temeli (projede kapsam dışı) |
| Cabrita vd., 2006 | Bakteriyel memetik algoritma (BMA) | Makalenin hibrit optimizer'ı |
| Wu vd., 2020 (MBGD-RDA) | TSK için mini-batch GD + düzenlileştirme + DropRule + AdaBound | Bulanık sistemleri sinir ağı teknikleriyle eğitmenin güncel örneği |
| Cui vd., 2021 | Yüksek boyutta TSK'nın başarısızlığı: softmax doygunluğu | Çok boyutlu veride kural ateşleme sorunları; bizim K^(1/M) genişlik ayarıyla aynı problem ailesi |

## 4. Kural başlatma ve kümelemeyle model tanımlama (katkıyla ilgili)

| Kaynak | Yöntem | Katkımızla ilişkisi |
|---|---|---|
| Bezdek vd., 1984 | Fuzzy C-Means | Önerilen `fcm` başlatmanın algoritması |
| Gustafson & Kessel, 1978 | Bulanık kovaryanslı kümeleme | Ürün uzayı kümelemenin klasik yöntemlerinden; `fcm_xy` ile aynı fikir ailesi |
| Chiu, 1994 | Küme kestirimi (subtractive clustering) ile bulanık model tanımlama | Kümelemeyle kural başlatmanın erken örneği; FCM'nin başlangıç değerleri için de kullanılıyor |
| Wang & Mendel, 1992 | Örneklerden kural üretme | Makale MEMS deneyinde başlatma için kullanıyor |
| **Shi vd., 2021 (FCM-RDpA)** | TSK regresyon modellerinde **ızgara başlatma yerine FCM** + gradyan eğitimi | **Katkımıza en yakın çalışma.** 22 veri setinde FCM başlatmanın ızgaradan iyi olduğunu, özellikle boyut arttıkça, gösteriyor |
| Glorot & Bengio, 2010 | Derin ağlarda başlatmanın eğitime etkisi | Başlatmanın gradyan tabanlı eğitimdeki önemine genel referans |

## 5. Yorumlanabilirlik

| Kaynak | Katkı | Projedeki rolü |
|---|---|---|
| Rudin, 2019 | Yüksek riskli kararlarda kara kutu açıklamak yerine doğası gereği yorumlanabilir model kullanma çağrısı | FRBN'nin motivasyonu |
| Lundberg & Lee, 2017 (SHAP) | Model-bağımsız öznitelik katkıları | Makale significance score'u SHAP ile karşılaştırıyor |

## 6. Literatürdeki yerimiz ve özgünlük sınırı

- **Kümelemeyle kural başlatma yeni bir fikir değil.** Chiu (1994), ürün uzayı kümelemesi ve özellikle Shi vd. (2021) bunu TSK sistemleri için gösteriyor. Raporda katkı bu yüzden "yeni bir algoritma" olarak değil, bilinen bir tekniğin **FRBN'ye uyarlanması ve etkisinin ölçülmesi** olarak konumlanıyor.
- **Bizim eklediklerimiz:**
  1. Referans makalenin raporlamadığı **başlatma duyarlılığının** nicel gösterimi (LM-TR için 20 başlatmada 9/20 yakınsama).
  2. Uyarlamanın **Mamdani + min t-norm + katmanlı** yapıya özgü gereksinimi: K^(1/M) genişlik ayarı ve sonuçların ateşleme-ağırlıklı başlatılması.
  3. Kazancın **kümelemeden mi kapsamadan mı** geldiğini ayıran ablation: `lhs` ile `fcm` karşılaştırması. Bulgumuz, Shi vd.'nin "FCM ızgaradan iyidir" sonucunu 1 ve 3 boyutlu FRBN için doğrulamıyor; düşük boyutta basit kantil ızgarası da aynı kazancı sağlıyor. Bu, iki çalışmanın boyut aralıklarının farklı olmasıyla tutarlı.
