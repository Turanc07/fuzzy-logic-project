"""Builds report/report.docx (Turkish technical report).

Numbers in the result tables are read from results/tables/*.csv, so the
report always matches the experiment outputs.

    python -m report.make_figures
    python -m report.build_report
    then export to PDF (Word: File > Save As > PDF, or report/export_pdf.ps1)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
F = ROOT / "results" / "figures"
OUT = ROOT / "report" / "report.docx"

# ----------------------------------------------------------------- references
REFS = {
    "lukacs": 'H.-I. Lukács, Á. Smidla, T. Fischl, and J. Botzheim, "Fuzzy rule-based network: a fuzzy logic-based interpretable and modular machine learning model," Scientific Reports, vol. 16, no. 1, Art. no. 34429, 2026, doi: 10.1038/s41598-026-63687-6.',
    "zadeh": 'L. A. Zadeh, "Fuzzy sets," Information and Control, vol. 8, no. 3, pp. 338–353, 1965, doi: 10.1016/S0019-9958(65)90241-X.',
    "mamdani": 'E. H. Mamdani and S. Assilian, "An experiment in linguistic synthesis with a fuzzy logic controller," International Journal of Man-Machine Studies, vol. 7, no. 1, pp. 1–13, 1975, doi: 10.1016/S0020-7373(75)80002-2.',
    "tsk": 'T. Takagi and M. Sugeno, "Fuzzy identification of systems and its applications to modeling and control," IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-15, no. 1, pp. 116–132, 1985, doi: 10.1109/TSMC.1985.6313399.',
    "ruspini": 'E. H. Ruspini, "A new approach to clustering," Information and Control, vol. 15, no. 1, pp. 22–32, 1969, doi: 10.1016/S0019-9958(69)90591-9.',
    "anfis": 'J.-S. R. Jang, "ANFIS: adaptive-network-based fuzzy inference system," IEEE Transactions on Systems, Man, and Cybernetics, vol. 23, no. 3, pp. 665–685, 1993, doi: 10.1109/21.256541.',
    "talpur": 'N. Talpur, S. J. Abdulkadir, H. Alhussian, M. H. Hasan, N. Aziz, and A. Bamhdi, "Deep neuro-fuzzy system application trends, challenges, and future perspectives: a systematic survey," Artificial Intelligence Review, vol. 56, no. 2, pp. 865–913, 2023, doi: 10.1007/s10462-022-10188-3.',
    "kerr": 'J. Kerr-Wilson and W. Pedrycz, "Generating a hierarchical fuzzy rule-based model," Fuzzy Sets and Systems, vol. 381, pp. 124–139, 2020, doi: 10.1016/j.fss.2019.07.013.',
    "huang": 'Y. Huang, D. Chen, W. Zhao, and Y. Lv, "Fuzzy C-means clustering based deep patch learning with improved interpretability for classification problems," IEEE Access, vol. 10, pp. 49873–49891, 2022, doi: 10.1109/ACCESS.2022.3171109.',
    "dcnfis": 'M. Yeganejou, K. Honari, R. Kluzinski, S. Dick, M. Lipsett, and J. Miller, "DCNFIS: Deep convolutional neuro-fuzzy inference system," arXiv:2308.06378, 2023.',
    "frdrl": 'W. Zhang, Z. Deng, G. Wang, and K.-S. Choi, "Fuzzy rule-based differentiable representation learning," IEEE Transactions on Neural Networks and Learning Systems, vol. 37, no. 2, pp. 603–616, 2026, doi: 10.1109/TNNLS.2025.3609722.',
    "mbgd": 'D. Wu, Y. Yuan, J. Huang, and Y. Tan, "Optimize TSK fuzzy systems for regression problems: Minibatch gradient descent with regularization, DropRule, and AdaBound (MBGD-RDA)," IEEE Transactions on Fuzzy Systems, vol. 28, no. 5, pp. 1003–1015, 2020, doi: 10.1109/TFUZZ.2019.2958559.',
    "cui": 'Y. Cui, D. Wu, and Y. Xu, "Curse of dimensionality for TSK fuzzy neural networks: Explanation and solutions," in Proc. Int. Joint Conf. Neural Networks (IJCNN), 2021, pp. 1–8, doi: 10.1109/IJCNN52387.2021.9534265.',
    "levenberg": 'K. Levenberg, "A method for the solution of certain non-linear problems in least squares," Quarterly of Applied Mathematics, vol. 2, no. 2, pp. 164–168, 1944, doi: 10.1090/qam/10666.',
    "marquardt": 'D. W. Marquardt, "An algorithm for least-squares estimation of nonlinear parameters," Journal of the Society for Industrial and Applied Mathematics, vol. 11, no. 2, pp. 431–441, 1963, doi: 10.1137/0111030.',
    "adam": 'D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," arXiv:1412.6980, 2014.',
    "bea": 'N. E. Nawa and T. Furuhashi, "Fuzzy system parameters discovery by bacterial evolutionary algorithm," IEEE Transactions on Fuzzy Systems, vol. 7, no. 5, pp. 608–616, 1999, doi: 10.1109/91.797983.',
    "bma": 'C. Cabrita, J. Botzheim, T. D. Gedeon, A. E. Ruano, L. T. Kóczy, and C. Fonseca, "Bacterial memetic algorithm for fuzzy rule base optimization," in Proc. World Automation Congress (WAC), 2006, pp. 1–6, doi: 10.1109/WAC.2006.376057.',
    "fcm": 'J. C. Bezdek, R. Ehrlich, and W. Full, "FCM: The fuzzy c-means clustering algorithm," Computers & Geosciences, vol. 10, no. 2–3, pp. 191–203, 1984, doi: 10.1016/0098-3004(84)90020-7.',
    "gk": 'D. Gustafson and W. Kessel, "Fuzzy clustering with a fuzzy covariance matrix," in Proc. IEEE Conf. Decision and Control, 1978, pp. 761–766, doi: 10.1109/CDC.1978.268028.',
    "chiu": 'S. L. Chiu, "Fuzzy model identification based on cluster estimation," Journal of Intelligent & Fuzzy Systems, vol. 2, no. 3, pp. 267–278, 1994, doi: 10.3233/IFS-1994-2306.',
    "wm": 'L.-X. Wang and J. M. Mendel, "Generating fuzzy rules by learning from examples," IEEE Transactions on Systems, Man, and Cybernetics, vol. 22, no. 6, pp. 1414–1427, 1992, doi: 10.1109/21.199466.',
    "fcmrdpa": 'Z. Shi, D. Wu, C. Guo, C. Zhao, Y. Cui, and F.-Y. Wang, "FCM-RDpA: TSK fuzzy regression model construction using fuzzy C-means clustering, regularization, Droprule, and Powerball Adabelief," Information Sciences, vol. 574, pp. 490–504, 2021, doi: 10.1016/j.ins.2021.05.084.',
    "glorot": 'X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," in Proc. 13th Int. Conf. Artificial Intelligence and Statistics (AISTATS), PMLR vol. 9, 2010, pp. 249–256.',
    "rudin": 'C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead," Nature Machine Intelligence, vol. 1, no. 5, pp. 206–215, 2019, doi: 10.1038/s42256-019-0048-x.',
    "shap": 'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Advances in Neural Information Processing Systems (NeurIPS), 2017, arXiv:1705.07874.',
}
ORDER: list[str] = []


def cite(*keys: str) -> str:
    nums = []
    for k in keys:
        if k not in REFS:
            raise KeyError(k)
        if k not in ORDER:
            ORDER.append(k)
        nums.append(ORDER.index(k) + 1)
    return "[" + ", ".join(str(n) for n in sorted(nums)) + "]"


# ----------------------------------------------------------------- document
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.2))
sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(2.0)

st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(4)
st.paragraph_format.line_spacing = 1.1
for lvl, size in ((1, 13), (2, 11.5)):
    h = doc.styles[f"Heading {lvl}"]
    h.font.name, h.font.size, h.font.bold = "Calibri", Pt(size), True
    h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    h.paragraph_format.space_before = Pt(10 if lvl == 1 else 6)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True

# page numbers in footer
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
    r = fp.add_run()
    if tag:
        el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag); r._r.append(el)
    else:
        el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = text; r._r.append(el)

fig_no = [0]
tab_no = [0]


def H(text: str, lvl: int = 1):
    doc.add_heading(text, level=lvl)


def P(text: str, italic: bool = False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size: float | None = None,
      highlight: bool = False):
    """Paragraph; **bold** segments supported."""
    p = doc.add_paragraph()
    p.alignment = align
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run(part)
        r.bold = i % 2 == 1
        r.italic = italic
        if size:
            r.font.size = Pt(size)
        if highlight:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return p


def B(items: list[str]):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        parts = it.split("**")
        for i, part in enumerate(parts):
            if part:
                p.add_run(part).bold = i % 2 == 1
        p.paragraph_format.space_after = Pt(2)


def EQ(text: str, num: int):
    t = doc.add_table(rows=1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.columns[0].width, t.columns[1].width = Cm(14.5), Cm(2)
    c0, c1 = t.rows[0].cells
    c0.width, c1.width = Cm(14.5), Cm(2)
    p = c0.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text); r.italic = True; r.font.name = "Cambria Math"
    q = c1.paragraphs[0]; q.alignment = WD_ALIGN_PARAGRAPH.RIGHT; q.add_run(f"({num})")


def FIG(path: Path, caption: str, width_cm: float = 16.0):
    fig_no[0] += 1
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(f"Şekil {fig_no[0]}. "); r.bold = True; r.font.size = Pt(9)
    r2 = c.add_run(caption); r2.font.size = Pt(9)
    return fig_no[0]


def shade(cell, hex_fill: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def TAB(caption: str, header: list[str], rows: list[list[str]], widths: list[float] | None = None, size: float = 8.5):
    tab_no[0] += 1
    c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.keep_with_next = True
    r = c.add_run(f"Tablo {tab_no[0]}. "); r.bold = True; r.font.size = Pt(9)
    r2 = c.add_run(caption); r2.font.size = Pt(9)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]; cell.text = ""
        run = cell.paragraphs[0].add_run(h); run.bold = True; run.font.size = Pt(size)
        shade(cell, "DCE6F1")
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            cell = t.rows[i].cells[j]; cell.text = ""
            parts = str(v).split("**")
            for k, part in enumerate(parts):
                if part:
                    rr = cell.paragraphs[0].add_run(part); rr.font.size = Pt(size); rr.bold = k % 2 == 1
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tab_no[0]


def e(v: float) -> str:
    return f"{v:.2e}"


# ----------------------------------------------------------------- data
e1 = pd.read_csv(T / "e1_sinc_summary.csv")
e2 = pd.read_csv(T / "e2_trig_summary.csv")
e1b = pd.read_csv(T / "e1b_init.csv")
p1 = pd.read_csv(T / "p1_convergence.csv"); p1["init"] = p1.config.str.split(" \\| ").str[1]
p2 = pd.read_csv(T / "p2_sinc_cv_folds.csv"); p2["init"] = p2.config.str.split(" \\| ").str[1]
p3 = pd.read_csv(T / "p3_trig_cv_folds.csv"); p3["init"] = p3.config.str.split(" \\| ").str[1]
SUCC = 1e-5

# ================================================================= TITLE
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Bulanık Kural Tabanlı Ağın (FRBN) Yeniden Gerçekleştirilmesi ve\nKapsama-Bilinçli Kural Başlatma ile Geliştirilmesi")
r.bold = True; r.font.size = Pt(15); r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
P("Turan ÇALDIRAN (1306220018) — Bulanık Mantık Dersi Dönem Projesi Teknik Raporu", align=WD_ALIGN_PARAGRAPH.CENTER, size=10.5)
P("Referans makale (No. 44): Lukács vd., Scientific Reports 16, 34429 (2026) · Kod: github.com/Turanc07/fuzzy-logic-project",
  align=WD_ALIGN_PARAGRAPH.CENTER, size=9)

H("Özet", 2)
P("Bu çalışmada, sinir ağlarının katmanlı yapısını Mamdani tipi bulanık kural tabanlarıyla birleştiren Bulanık Kural Tabanlı Ağ "
  f"(Fuzzy Rule-Based Network, FRBN) {cite('lukacs')} sıfırdan yeniden gerçekleştirilmiş, makalenin iki temel deneyi (sinc fonksiyonu "
  "ve üç boyutlu trigonometrik veri) 5-katlı çapraz doğrulamayla tekrarlanmıştır. Yeniden gerçekleştirme, makalenin sinc için "
  "raporladığı 10⁻⁷–10⁻⁸ mertebesindeki hataya ulaşabildiğini doğrulamış; ancak makalede raporlanmayan güçlü bir başlatma "
  "duyarlılığı ortaya çıkarmıştır: Levenberg–Marquardt (trust-region) ile eğitilen FRBN, 20 rastgele başlatmanın yalnızca 9'unda "
  "yakınsamaktadır. Nedenin, rastgele başlatmanın kuralları veri üzerine dengesiz dağıtması (aralık dışında kalan, üst üste yığılan "
  "ya da hiç ateşlenmeyen kurallar) ve yerel bir yöntem olan LM'nin bunu düzeltememesi olduğu gösterilmiştir. Katkı olarak ilk katmanın kurallarını eğitim verisinden yerleştiren kapsama-bilinçli "
  "bir başlatma (Fuzzy C-Means tabanlı ve kantil ızgarası tabanlı) önerilmiştir. Önerilen başlatma yakınsama oranını 9/20'den 20/20'ye "
  "çıkarmış (Fisher p < 0,001), sinc'te altı konfigürasyonun altısında ortalama hatayı düşürmüş ve trigonometrik veride ölü kuralları "
  "ortadan kaldırmıştır. Kümeleme yapmayan kontrol yönteminin de benzer kazanç sağlaması, iyileşmenin asıl kaynağının kümeleme değil "
  "kapsama olduğunu göstermektedir.", size=10)

# ================================================================= 1
H("1. Giriş (Introduction)")
P("Derin sinir ağları birçok alanda yüksek doğruluk sağlasa da iç mantıkları okunamayan \"kara kutu\" modellerdir. Yüksek riskli "
  f"kararlarda bu modellerin sonradan açıklanmaya çalışılması yerine doğası gereği yorumlanabilir modellerin kullanılması önerilmektedir {cite('rudin')}. "
  f"Bulanık mantık {cite('zadeh')} ve Mamdani tipi kural tabanlı sistemler {cite('mamdani')}, bilgiyi \"EĞER … İSE …\" biçiminde dilsel "
  f"kurallarla temsil ettikleri için bu ihtiyaca doğal bir cevaptır. ANFIS {cite('anfis')} gibi nöro-bulanık modeller kural parametrelerini "
  "veriden öğrenebilir; ancak tek katmanlıdır ve giriş sayısı arttıkça kural sayısı hızla büyür.")
P(f"Referans makale {cite('lukacs')}, nöronlar yerine bulanık kural katmanlarından oluşan ve derinleştirilebilen bir \"gri kutu\" "
  "model olan FRBN'yi önermektedir. Her katman bir bulanık kural tabanıdır; katmanlar zincirlenerek derin bir yapı oluşturulur. "
  "Makale; Adam, Levenberg–Marquardt ve bakteriyel memetik algoritma ile eğitilen FRBN'nin sinc fonksiyonunda tek gizli katmanlı "
  "sinir ağıyla rekabetçi olduğunu, kural önem skoru ve kapsama oranı gibi ölçütlerle modüler ve yorumlanabilir olduğunu göstermektedir.")
P("Bu projenin amaçları şunlardır: (i) FRBN'yi resmi koddan bağımsız olarak yeniden gerçekleştirmek; (ii) makaledeki sinc ve "
  "trigonometrik deneyleri tekrarlayıp sonuçları karşılaştırmak; (iii) yeniden gerçekleştirmede ortaya çıkan bir zayıflığa yönelik "
  "teknik bir katkı geliştirip kontrollü deneylerle değerlendirmek. Makaledeki endüstriyel MEMS veri seti (yaklaşık 800 bin kayıt, "
  "anonimleştirilmiş) kapsam dışında bırakılmıştır.")

# ================================================================= 2
H("2. İlgili Çalışmalar (Related Work)")
P(f"**Nöro-bulanık ve derin bulanık modeller.** ANFIS {cite('anfis')}, TSK {cite('tsk')} tipi bir bulanık sistemi sinir ağı yapısında "
  f"öğrenen klasik yöntemdir. Talpur vd. {cite('talpur')} 2015–2020 arasındaki derin nöro-bulanık sistemleri sistematik olarak "
  f"inceleyerek yorumlanabilirlik ile makul doğruluk arasındaki dengeyi ve araştırma boşluklarını özetlemektedir. Kerr-Wilson ve "
  f"Pedrycz {cite('kerr')} kademeli (hiyerarşik) bir TSK kural tabanını veriden çıkarmakta; Huang vd. {cite('huang')} derin "
  f"\"patch learning\" sınıflandırıcılarında ızgara bölüntüsünün yol açtığı kural patlamasını Fuzzy C-Means (FCM) ile önlemektedir. "
  f"DCNFIS {cite('dcnfis')} evrişimli ağları bulanık çıkarımla birleştirirken, FRDRL {cite('frdrl')} TSK öncül kısmını türevlenebilir "
  "bir temsil öğrenme katmanı olarak kullanmaktadır. FRBN bu çalışmalardan, Mamdani kural katmanlarını doğrudan zincirlemesi ve "
  "her katmanda dilsel olarak okunabilir kurallar tutmasıyla ayrılmaktadır.")
P(f"**Bulanık sistemlerin eğitimi.** Wu vd. {cite('mbgd')} TSK sistemlerini mini-batch gradyan inişi, düzenlileştirme ve DropRule ile "
  f"eğitmenin büyük veride etkili olduğunu göstermiştir. Cui vd. {cite('cui')}, Gauss üyelik fonksiyonlu TSK modellerinin yüksek "
  "boyutta kural ateşleme güçlerinin küçülmesi nedeniyle başarısız olduğunu açıklamıştır. Referans makale ise sönümlü Gauss–Newton "
  f"yöntemi Levenberg–Marquardt {cite('levenberg', 'marquardt')}, Adam {cite('adam')} ve bakteriyel evrimsel/memetik algoritmaları "
  f"{cite('bea', 'bma')} kullanmaktadır.")
P(f"**Kümelemeyle kural başlatma.** Kural tabanını kümelemeyle başlatmak bulanık model tanımlamada yerleşik bir yaklaşımdır: FCM {cite('fcm')}, "
  f"bulanık kovaryanslı Gustafson–Kessel kümelemesi {cite('gk')} ve küme kestirimi {cite('chiu')} bu amaçla kullanılmaktadır. "
  f"Wang–Mendel yöntemi {cite('wm')} kuralları doğrudan örneklerden üretir; referans makale MEMS deneyinde başlatma için bu yöntemi "
  f"kullanmaktadır. Katkımıza en yakın çalışma Shi vd.'nin FCM-RDpA yöntemidir {cite('fcmrdpa')}: TSK regresyon modellerinde ızgara "
  "bölüntüsüyle başlatmayı FCM ile değiştirmiş ve 22 veri setinde, özellikle boyut arttıkça daha iyi sonuç elde etmiştir. Dolayısıyla "
  "kümelemeyle başlatma tek başına yeni bir fikir değildir. Bu projenin katkısı, (i) referans makalenin raporlamadığı başlatma "
  "duyarlılığını nicel olarak göstermek, (ii) tekniği Mamdani + min t-norm + katmanlı FRBN yapısına uyarlamak ve (iii) kazancın "
  f"kümelemeden mi kapsamadan mı geldiğini ayıran bir kontrol deneyi yapmaktır. Başlatmanın gradyan tabanlı eğitimdeki genel önemi "
  f"derin ağlar için de bilinmektedir {cite('glorot')}.")

# ================================================================= 3
H("3. Teorik Altyapı (Theoretical Background)")
P(f"Bir bulanık küme A, evrensel küme U üzerinde μ_A: U → [0, 1] üyelik fonksiyonuyla tanımlanır {cite('zadeh')}. Klasik kümelerde "
  "bir eleman kümeye ya aittir ya da değildir; bulanık kümelerde ise kısmi üyelik vardır ve bu, \"sıcaklık yüksek\" gibi keskin "
  "sınırı olmayan kavramların modellenmesini sağlar. Projede iki üyelik fonksiyonu kullanılmıştır. Dört kırılma noktalı (a ≤ b ≤ c ≤ d) "
  "trapezoid üyelik fonksiyonu")
EQ("μ(x) = max( min( (x − a)/(b − a), 1, (d − x)/(d − c) ), 0 )", 1)
P("biçimindedir; çekirdeği (üyeliğin 1 olduğu bölge) [b, c], desteği (üyeliğin 0'dan büyük olduğu bölge) (a, d) aralığıdır. Gauss "
  "üyelik fonksiyonu μ(x) = exp(−(x − μ)²/(2σ²)) ise iki parametrelidir; çekirdeği tek noktadır ({μ}) ve pratik destek μ ± 3σ alınır. "
  "Trapezoidin düz bölgelerinde (çekirdekte ve destek dışında) türev sıfırdır; Gauss ise her yerde türevlenebilir ve sıfırdan "
  "farklıdır. Bu fark, gradyan tabanlı eğitimde iki tipin farklı davranmasının temel nedenidir. Dilsel bir değişken (ör. \"x düşük\", "
  "\"x orta\", \"x yüksek\") bu kümelerle ifade edilir.")
P("Mamdani tipi bir kural tabanında kural i, \"EĞER x₁, Aᵢ₁ VE … VE x_M, Aᵢ_M İSE y, Cᵢ\" biçimindedir. Bulanıklaştırma adımında her giriş "
  "için üyelik derecesi λᵢⱼ(xⱼ) hesaplanır; VE bağlacı bir t-norm ile, FRBN'de minimum ile gerçeklenir:")
EQ("wᵢ(x) = min_{j=1..M} λᵢⱼ(xⱼ)", 2)
P("Minimum t-norm'un önemli bir sonucu vardır: kuralın ateşlenmesi için örneğin her boyutta o kuralın desteğinde olması gerekir ve "
  "wᵢ'nin parametrelere göre gradyanı yalnızca en küçük üyeliği veren öncüle akar. Her kuralın sonuç kümesi ateşleme gücüyle kesilir "
  "ve kümeler birleştirilir. Durulaştırma için trapezoid sonuçlarda Center of Sums (CoS) yöntemi kullanılır: kesilmiş trapezoidlerin "
  "alanları ve momentleri ayrı ayrı toplanır, çıktı toplam momentin toplam alana oranıdır:")
EQ("y(x) = Σᵢ ∫ y · min(μ_Cᵢ(y), wᵢ) dy  /  Σᵢ ∫ min(μ_Cᵢ(y), wᵢ) dy", 3)
P("Yüksekliği w ile kesilen (a, b, c, d) trapezoidinin alanı w(d − a) + (w²/2)(c + a − d − b) olduğundan, momenti de dahil olmak üzere "
  "(3) kapalı biçimde hesaplanabilir; bu sayede durulaştırma türevlenebilir ve hızlıdır. Gauss sonuçlarda çıktı, sonuç merkezlerinin "
  "ateşleme güçleriyle ağırlıklı ortalamasıdır:")
EQ("y(x) = Σᵢ wᵢ(x) μ̂ᵢ / (Σᵢ wᵢ(x) + ε)", 4)
P("(4)'ten görüldüğü gibi bir x için hiçbir kural ateşlenmezse (tüm wᵢ = 0) çıktı sıfıra düşer ve hiçbir parametreye gradyan akmaz. "
  "Bu gözlem, Bölüm 6'daki katkının çıkış noktasıdır.")
P(f"TSK modellerinde {cite('tsk')} sonuç kısmı girişlerin bir fonksiyonudur; Mamdani modellerinde ise bulanık kümedir ve bu nedenle "
  "dilsel olarak daha doğrudan okunabilir. FCM kümelemesi, n örneği c kümeye üyelik dereceleri uᵢₖ ∈ [0, 1] ile atar ve "
  f"Σᵢ Σₖ uᵢₖᵐ ‖xᵢ − vₖ‖² amacını, merkezler vₖ ile üyelikler arasında dönüşümlü güncellemeyle en küçükler {cite('fcm')}. "
  "Bulanıklaştırıcı m > 1 (bu projede m = 2) kümelerin ne kadar örtüşeceğini belirler. Levenberg–Marquardt yöntemi, model "
  "çıktısının parametrelere göre Jacobian'ı J ve artık vektörü e ile")
EQ("s = −(JᵀJ + γI)⁻¹ Jᵀe", 5)
P(f"adımını atar; γ büyüdükçe küçük adımlı gradyan inişine, küçüldükçe hızlı yakınsayan Gauss–Newton yöntemine yaklaşır "
  f"{cite('levenberg', 'marquardt')}. Trust-region sönümlemesinde γ, gerçekleşen hata azalmasının doğrusal modelin öngördüğü azalmaya "
  "oranına göre ayarlanır:")
EQ("r = (‖e(b)‖² − ‖e(b + s)‖²) / (‖e(b)‖² − ‖Js + e(b)‖²)", 6)
P("r < 0,25 ise γ dört katına çıkarılır, r > 0,75 ise yarıya indirilir. Makalede (6)'nın paydasında hata yerine parametre normu "
  "yazılmıştır; bunu bir yazım hatası olarak değerlendirdik ve standart tanımı kullandık.")

# ================================================================= 4
H("4. Orijinal Yöntem (Original Method)")
fa = FIG(F / "arch.png", "FRBN mimarisi. Her katman, paylaşılan öncüller ve çıkış boyutu başına bir sonuç kümesi içeren K kurallık bir kural tabanıdır.", 15.5)
P(f"FRBN, [M, N, K] ile tanımlanan bulanık katmanların zincirlenmesidir: M giriş, N çıkış ve K kural (Şekil {fa}). Katmanda K×M "
  "boyutlu bir öncül matrisi ile K×N boyutlu bir sonuç matrisi bulunur; N çıkışın her biri için öncülleri paylaşan ayrı bir kural "
  "tabanı kurulur. Bu nedenle katman doğal olarak çok girişli ve çok çıkışlıdır. Ağın çıktısı y = f_L(…f₂(f₁(x))) biçimindedir ve "
  "doğrusal olmama üyelik fonksiyonlarından geldiği için katmanlar arasında ek aktivasyon fonksiyonu kullanılmaz. Eğitilen "
  "parametreler kırılma noktaları (trapezoid) ya da (μ, σ) değerleridir (Gauss).")
P("Makale eğitim için Adam, standart ve trust-region sönümlemeli Levenberg–Marquardt (LM-TR) ve bunların bakteriyel evrimsel algoritmayla "
  "birleştiği hibrit bakteriyel memetik algoritmayı (BMA) kullanır. LM-TR'de sönüm katsayısı, gerçekleşen ve öngörülen hata azalması "
  "oranı r'ye göre güncellenir (r < 0,25 ise 4γ, r > 0,75 ise γ/2) ve yalnızca hatayı azaltan adımlar kabul edilir. Yorumlanabilirlik "
  "için iki ölçüt önerilir: bir kuralın belirli bir girdi için çıktıya göreli katkısını veren önem skoru (significance score) ve kural "
  "desteğinin giriş uzayını ne kadar kapladığını ölçen minimum öncül kapsama oranı ρᵢ (1'e yakınsa global, 0'a yakınsa yerel kural).")
P("Makalenin raporladığı başlıca sonuçlar şunlardır: sinc'te 20 kurallı Gauss FRBN, LM-TR ile ortalama 8,07×10⁻⁷ ± 5,78×10⁻⁷ "
  "doğrulama hatasına (en iyi fold 6,2×10⁻⁸) ulaşmakta ve 20 nöronlu tanh ağıyla rekabet etmektedir; ancak sinir ağı eğitimi daha "
  "kararlıdır. Trigonometrik veride 7 kurallı [3,1,7] trapezoid model en iyi mimari olarak seçilmiş, LM-TR ortalama 1,09×10⁻² ile "
  "en iyi optimizer olarak raporlanmış ve modelin gürültü özelliği x₁'i yok saydığı belirtilmiştir.")

# ================================================================= 5
H("5. Yeniden Gerçekleştirme Yöntemi (Reproduction Methodology)")
P("Tüm bileşenler PyTorch ile bu proje kapsamında yazılmıştır; resmi depodaki kod kopyalanmamış, yalnızca makalede yazmayan "
  "ayrıntılar (sinc aralığı, başlatma aralıkları) için okunmuştur. Bileşenler ve doğrulama yolları şunlardır:")
B(["**Üyelik fonksiyonları ve katman** (src/membership.py, src/fuzzy_layer.py): Trapezoid parametreleri her ileri geçişte sıralanarak "
   "a ≤ b ≤ c ≤ d kısıtı korunur. CoS kapalı formülü, kesilmiş trapezoidlerin sayısal integraliyle dört farklı durumda 10⁻⁴ "
   "hassasiyetle doğrulanmıştır.",
   "**Ağ** (src/fuzzy_network.py): Katman zincirleme, parametre sayımı ve kuralların \"EĞER … İSE …\" biçiminde yazdırılması. "
   "[3,1,7] için 112 ve [3,1,7]-[1,1,7]-[1,1,4] için 200 parametre, makalenin Tablo 2'siyle aynıdır.",
   "**Optimizer'lar** (src/optim_lm.py, src/train.py): Adam ve LM / LM-TR bağımsız olarak yazılmıştır (makale harici bir kütüphane "
   "kullanır). Jacobian, örnek başına gradyanların vektörleştirilmesiyle (torch.func.vmap + jacrev) hesaplanır. LM'nin doğrusal en "
   "küçük kareler problemini tam çözdüğü ve hatayı hiçbir adımda artırmadığı birim testleriyle doğrulanmıştır.",
   "**Karşılaştırma modeli** (src/nn_baseline.py): 1-20-1 MLP, makaledeki ile aynı 61 parametre.",
   "**Deney altyapısı** (src/experiment.py): 5-katlı çapraz doğrulama, iki başlatma protokolü, tablo ve grafik üretimi. Projede "
   "toplam 47 birim testi bulunmaktadır.",
   "**Farklı giriş değerleriyle test** (demo/frbn_demo.py): Eğitilmiş sistem, verilen herhangi bir girdi için üyelik derecelerini, "
   "min t-norm ile ateşleme güçlerini, her kuralın payını ve durulaştırılmış çıktıyı gerçek değerle birlikte gösterir. Örneğin "
   "trigonometrik modelde x = (0,2; 0,15; 0,5) için yalnızca iki kural ateşlenmekte, çıktı 0,132 ve gerçek değer 0,096 olmaktadır; "
   "veri aralığı dışındaki bir girdide (x = (5; 5; 5)) hiçbir kural ateşlenmemekte ve çıktı 0'a düşmektedir."])
P("Makale ile resmi kod arasında bazı farklar bulunmuş ve makale esas alınmıştır: sinc için 20.000 nokta (notebook 200.000 kullanıyor) "
  "ve 5-katlı çapraz doğrulama (yayımlanan notebook'ta yok). Ayrıca makalede bazı tutarsızlıklar tespit edilmiştir: Tablo 1(a)'nın "
  "başlığı \"NN\" dediği halde FN sonuçlarını içermekte; ablation modeli için 40 parametre verilmekte (doğrusu 60, sonuç kümeleri "
  "sayılmamış); MEMS bölümünde metin BMA'yı en iyi ilan ederken tablo GradOnly LM-TR'yi en iyi göstermektedir.")
P("Resmi notebook her konfigürasyondan önce aynı tohumu (42) ayarladığı için tüm fold'lar aynı başlangıç ağırlıklarıyla eğitilmektedir. "
  "Bu nedenle iki protokol kullanılmıştır: makale protokolü (tüm fold'larda tohum 42) ve değişken protokol (her fold'da farklı tohum). "
  "Böylece makaledeki düşük standart sapmanın başlatma varyansını içerip içermediği ölçülebilmiştir.")

# ================================================================= 6
H("6. Önerilen Geliştirme (Proposed Extension)")
P("**Gözlenen problem.** Yeniden gerçekleştirme (Bölüm 8.1), FRBN'nin LM ile eğitiminin başlatmaya çok duyarlı olduğunu göstermiştir. "
  "Rastgele başlatmada (trapezoid kırılma noktaları ~ U(−1, 2), Gauss (μ, σ) ~ U(−1, 1)) kurallar verinin bulunduğu aralığa göre "
  "yerleştirilmez: bazı kurallar aralığın dışında kalır, bazıları üst üste yığılır, bazı bölgeler yalnızca çok geniş kurallarla "
  "temsil edilir (Şekil 2'nin solu). İki mekanizma bu durumun eğitimle düzeltilmesini zorlaştırır. Birincisi, wᵢ = 0 olan bir kuralın "
  "parametrelerinin o örneklere göre gradyanı sıfırdır; hiç ateşlenmeyen bir kural (ölü kural) ya da hiçbir kuralın ateşlenmediği "
  "bir bölge gradyanla kurtarılamaz. Min t-norm gradyanı yalnızca en küçük üyeliğe aktarır ve trapezoidin düz bölgelerinde gradyan "
  "zaten sıfırdır. İkincisi, LM yerel bir yöntemdir ve kuralları başlangıç konumlarından uzağa taşıyamaz. Trigonometrik veride "
  "birinci mekanizma (ölü kural), sinc'te ikincisi (dengesiz kural dağılımı) gözlenmiştir.")
fi = FIG(F / "init_mfs.png", "Eğitimden önce 20 kuralın öncül üyelik fonksiyonları. Solda rastgele başlatma: kurallar üst üste yığılıyor, bir kısmı veri aralığının dışına taşıyor. Sağda FCM tabanlı başlatma: kurallar veri aralığına düzgün yayılıyor.")
P(f"**Önerilen yöntem.** İlk katmanın öncül kümeleri eğitim verisinden yerleştirilir (src/initialization.py; Şekil {fi}). Üç varyant "
  "karşılaştırılmıştır:")
B(["**fcm (önerilen):** Girişler FCM ile K kümeye ayrılır; küme k, kural k'nın merkezini verir. Genişlik, kümenin bulanık standart "
   "sapmasıdır (√(Σᵢ uᵢₖ²(xᵢ − vₖ)² / Σᵢ uᵢₖ²)).",
   f"**fcm_xy (varyant):** FCM, girişler ve çıkış birlikte, yani ürün uzayında [x, y] çalıştırılır; merkezler girişe izdüşürülür. Bu, "
   f"klasik ürün uzayı kümelemesiyle model tanımlama yaklaşımıdır {cite('gk', 'chiu')}.",
   "**lhs (kontrol):** Kümeleme yapılmaz; her boyutta K kural merkezi, verinin (k + 0,5)/K kantillerine yerleştirilir ve boyutlar arası "
   "eşleştirme rastgele permütasyonla yapılır (Latin hiperküpü). Bu kontrol, kazancın kümelemeden mi yoksa yalnızca kapsamadan mı "
   "geldiğini ayırt etmek için eklenmiştir."])
P("Min t-norm altında bir örneğin bir kuralı ateşlemesi için her boyutta o kuralın desteğinde olması gerekir. K kural M boyutta boyut "
  "başına K^(1/M) ızgara hücresine karşılık geldiğinden, en küçük genişlik bu hücre boyunun yarısıyla sınırlanmıştır. Bu ayar "
  "yapılmadığında 3 boyutlu trigonometrik veride örneklerin %38'i hiçbir kuralla kapsanmıyordu. Trapezoidin çekirdeği merkez ± 0,5w, "
  "desteği merkez ± 2w olarak alınır. Sonuç kümeleri, yeni öncüllerle hesaplanan ateşleme güçleriyle ağırlıklandırılmış y ortalaması "
  "ve yayılımına göre ayarlanır; böylece model eğitime kabaca doğru bir parçalı yaklaşımla başlar. Mimari, parametre sayısı, çıkarım "
  "ve optimizer değişmez; yalnızca başlangıç noktası değişir.")

# ================================================================= 7
H("7. Deney Düzeni (Experimental Setup)")
P("**Veri.** sinc: x ∈ [−1, 1] aralığında eşit aralıklı 20.000 nokta, y = sin(10πx)/(10πx). Trigonometrik: x₂, x₃ ~ U[0, 1], gürültü "
  "özelliği x₁ ~ U[0, 1], 8000 örnek, y = (sin(2πx₂)·cos(2πx₃) + 1)/2. Makale y'nin [0, 1]'e normalize edildiğini belirtmekte ancak "
  "dönüşümü vermemektedir; bu dönüşüm bizim varsayımımızdır. Veri setleri kodla üretildiği için repoda ayrıca saklanmamıştır.")
P("**Yazılım ve donanım.** Python 3.14.5, PyTorch 2.15.0 (CPU), NumPy 2.5.4, pandas 3.1.1, SciPy 1.18.1; AMD Ryzen 7 6800H işlemcili, "
  "16 GB bellekli Windows 11 dizüstü bilgisayar. Tüm sürümler requirements.txt dosyasında sabitlenmiştir.")
P("**Eğitim.** 100 epoch; her epoch eğitim kümesinin onda biri büyüklüğünde 10 mini-batch'ten oluşur (toplam 1000 optimizer adımı, "
  "resmi koddaki oran). Adam için öğrenme oranı 0,01, LM için γ₀ = 1. Mimariler: sinc için [1,1,20], trigonometrik veri için [3,1,7]. "
  "Karşılaştırma ağı: 1-20-1 MLP.")
P("**Deneyler.** E1: sinc, FN ve NN, üç optimizer, iki protokol, 5-katlı CV. E1b: aynı veri bölmesinde 20 farklı başlatma tohumu. "
  "E2: trigonometrik veri, [3,1,7] trapezoid FN. P1: E1b protokolü, dört başlatma yöntemi × üç optimizer × 20 tohum. P2: sinc, "
  "5-katlı CV, iki üyelik tipi × üç optimizer × dört başlatma. P3/P4: trigonometrik veri, 5 fold × 2 başlatma (10 eşleştirilmiş çift) "
  "ve yorumlanabilirlik ölçütleri. P5: başlatma ve eğitim süreleri. P6: eğitilmiş modelin kurallarının dilsel etiketlere çevrilmesi "
  "ve gerçek veriyle tutarlılığının kontrolü.")
P("**Ölçütler ve istatistik.** Doğrulama MSE (ortalama ± std, medyan, en iyi), yakınsama oranı (doğrulama MSE < 10⁻⁵), eğitim "
  "sonunda hiç ateşlenmeyen (ölü) kural sayısı, hiçbir kuralın ateşlenmediği doğrulama örneği oranı, ortalama kapsama oranı ρ ve süre. "
  "Önerilen yöntemler, aynı tohum ve fold'larla eşleştirilmiş Wilcoxon işaretli sıra testi (log₁₀ MSE, çift yönlü) ve yakınsama "
  "sayıları için Fisher kesin testiyle rastgele başlatmaya karşı test edilmiştir (α = 0,05). Başlatma yalnızca eğitim fold'undan "
  "hesaplanır; doğrulama verisi sızmaz.")

# ================================================================= 8
H("8. Sonuçlar (Results)")
H("8.1 Yeniden gerçekleştirme", 2)


def e1row(cfg: str) -> list[str]:
    pa = e1[(e1.config == cfg) & (e1.protocol == "paper")].iloc[0]
    va = e1[(e1.config == cfg) & (e1.protocol == "varied")].iloc[0]
    return [cfg.replace("_", "-").replace("lm-tr", "LM-TR").replace(" lm", " LM").replace(" adam", " Adam"),
            f"{e(pa.val_mean)} ± {e(pa.val_std)}", f"{e(va.val_mean)} (en iyi {e(va.val_best)})",
            f"{e(pa.paper_val_mean)} ± {e(pa.paper_val_std)}"]


t1 = TAB("E1 — sinc, 5-katlı CV doğrulama MSE: bizim sonuçlarımız ve makale (Tablo 1).",
         ["Konfigürasyon", "Bizim (makale protokolü)", "Bizim (değişken başlatma)", "Makale"],
         [e1row(c) for c in ["FN gauss lm_tr", "FN gauss lm", "FN gauss adam", "FN trap lm_tr", "FN trap adam",
                             "NN tanh lm_tr", "NN tanh lm", "NN relu lm_tr"]], [4.0, 4.2, 4.6, 3.8])
P(f"Tablo {t1}, sinir ağı tarafının makaleyle örtüştüğünü göstermektedir: tanh + LM-TR en iyi sinir ağı konfigürasyonudur "
  f"({e(e1[(e1.config=='NN tanh lm_tr')&(e1.protocol=='paper')].val_mean.iloc[0])}, makalede 2,82×10⁻⁷). Bu, LM uygulamasının doğru "
  "olduğunu gösterir. FRBN de makaledeki seviyeye ulaşabilmektedir: Gauss + LM-TR ile değişken protokolde en iyi fold "
  f"{e(e1[(e1.config=='FN gauss lm_tr')&(e1.protocol=='varied')].val_best.iloc[0])} bulunmuştur. Trapezoid FN'nin Gauss FN'den kötü "
  "olması da makaleyle tutarlıdır. Buna karşın makale protokolünde FN Gauss + LM ve LM-TR beş fold'un hepsinde ~4×10⁻³'te "
  "takılmıştır.")
fe1 = FIG(F / "e1_sinc_preds_fn.png", "E1 — Makale protokolünde (tohum 42) fold 0 için FN tahminleri. Gauss + LM-TR modeli x ≈ ±0,1–0,3 aralığındaki yan dalgalanmaları öğrenemiyor.", 14.0)
P(f"Şekil {fe1}'te bu modelin ana tepeyi ve uç bölgeleri doğru öğrendiği, ancak x ≈ ±0,1–0,3 aralığındaki yan dalgalanmaları tamamen "
  "kaçırdığı görülmektedir. Aynı başlatmayla Adam bu dalgalanmaları bulabilmektedir; dolayısıyla sorun modelin kapasitesinde değil, "
  "LM'nin başlangıç noktasındadır. Eğitilmiş modelin kuralları incelendiğinde o bölgede kuralların ateşlendiği, ancak bunların "
  "dalgalanmayı temsil edemeyecek kadar geniş olduğu görülmektedir: 0,1–0,3 aralığında merkezi bulunan tek kuralın genişliği "
  "σ ≈ 1,1'dir. 20 kural merkezinin üçü veri aralığının dışında (≈ 1,07–1,10), ikisi aynı noktada (0,0) kalmıştır. Yani sorun, "
  "kuralların veri üzerine dengesiz dağılmasıdır ve yerel bir yöntem olan LM kuralları uzak bölgelere taşıyamamaktadır.")
e1b_s = e1b.groupby("config", sort=False).val_mse.agg(lambda v: (np.asarray(v) < SUCC).sum())
t2 = TAB("E1b — Aynı veri bölmesinde 20 farklı başlatma (sinc, fold 0).",
         ["Konfigürasyon", "Yakınsayan (< 10⁻⁵)", "Medyan", "En iyi", "En kötü"],
         [[c.replace("lm_tr", "LM-TR").replace(" lm", " LM"), f"{int(e1b_s[c])}/20",
           e(e1b[e1b.config == c].val_mse.median()), e(e1b[e1b.config == c].val_mse.min()), e(e1b[e1b.config == c].val_mse.max())]
          for c in e1b.config.unique()], [4.2, 3.4, 2.8, 2.8, 2.8])
P(f"Tablo {t2} bu duyarlılığı nicel olarak göstermektedir: FN + LM-TR yalnızca 9/20, FN + LM 13/20 başlatmada yakınsarken aynı "
  "parametre bütçesindeki sinir ağı 20/20'de yakınsamaktadır. FN sonuçları iki kümeye ayrılır: çalıştırmalar ya ~10⁻⁷ seviyesine "
  "iner ya da ~10⁻³'te takılır. Makaledeki düşük standart sapma, tüm fold'larda aynı başlatmanın kullanılmasından kaynaklanıyor "
  "olabilir.")
e2r = []
for opt, lab in (("adam", "Adam"), ("lm", "LM"), ("lm_tr", "LM-TR")):
    pa = e2[(e2.config == f"FN trap {opt}") & (e2.protocol == "paper")].iloc[0]
    va = e2[(e2.config == f"FN trap {opt}") & (e2.protocol == "varied")].iloc[0]
    e2r.append([lab, f"{e(pa.val_mean)} ± {e(pa.val_std)}", f"{e(va.val_mean)} ± {e(va.val_std)}", f"{e(pa.paper_val_mean)} ± {e(pa.paper_val_std)}"])
t3 = TAB("E2 — Trigonometrik veri, [3,1,7] trapezoid FN, doğrulama MSE (makale Tablo 3).",
         ["Optimizer", "Bizim (makale protokolü)", "Bizim (değişken)", "Makale"], e2r, [2.6, 4.6, 4.6, 4.6])
P(f"Trigonometrik veride (Tablo {t3}) Adam makaleden iyi, LM ve LM-TR ise yaklaşık 2,5 kat kötüdür; makalenin \"LM-TR en iyi\" "
  "sıralaması tekrarlanamamıştır. Olası nedenler: y normalizasyonunun belirsizliği, makalenin kullandığı farklı LM kütüphanesi ve "
  "başlatma. Tüm veriyle eğitilen modelde yedi kuraldan biri hiç ateşlenmemektedir (ölü kural). Gürültü özelliği x₁ ise yedi kuralın "
  "yalnızca ikisinde tamamen yok sayılmaktadır; makaledeki iddia kısmen doğrulanmıştır.")

H("8.2 Önerilen başlatmanın değerlendirilmesi", 2)
lab = {"random": "random (baseline)", "lhs": "lhs (kontrol)", "fcm": "fcm (önerilen)", "fcm_xy": "fcm_xy (varyant)"}
rows = []
for opt, olab in (("lm_tr", "LM-TR"), ("lm", "LM"), ("adam", "Adam")):
    rows.append([olab] + [f"{int((p1[(p1.optimizer==opt)&(p1.init==i)].val_mse < SUCC).sum())}/20" for i in lab])
t4 = TAB("P1 — Yakınsama oranı (sinc, Gauss [1,1,20], fold 0, 20 başlatma; val. MSE < 10⁻⁵).",
         ["Optimizer"] + list(lab.values()), rows, [2.4, 3.4, 3.2, 3.4, 3.4])
fp1 = FIG(F / "p1_convergence.png", "P1 — Başlatma yöntemine göre 20 çalıştırmanın son doğrulama hatası (her nokta bir başlatma; noktalı çizgi yakınsama eşiği).", 16.5)
P(f"Tablo {t4} ve Şekil {fp1}, önerilen başlatmanın baseline'daki iki kümeli dağılımı ortadan kaldırdığını göstermektedir. LM-TR ile "
  "yakınsama 9/20'den 20/20'ye (fcm ve lhs; Fisher p < 0,001), LM ile 13/20'den 19/20'ye (fcm; p = 0,044), Adam ile 2/20'den 10/20'ye "
  "(fcm; p = 0,014) çıkmaktadır. LM-TR'de medyan hata 1,98×10⁻³'ten 1,05×10⁻⁷'ye düşmüştür (Wilcoxon p < 0,001). Medyan başlangıç "
  "hatası da rastgele başlatmada 6,4×10⁻², fcm'de 1,6×10⁻²'dir; yani model eğitime kabaca doğru bir yaklaşımla başlamaktadır. "
  "fcm_xy daha zayıftır (LM-TR 17/20).")
fp2 = FIG(F / "p2_sinc_preds.png", "P2 — Gauss FN + LM-TR, fold 0: başlatma yöntemine göre eğitim sonrası tahminler. fcm_xy, uçlardaki düşük genlikli dalgalanmalarda düz bölgeler bırakıyor.", 14.0)
P(f"Şekil {fp2} bu farkın nedenini göstermektedir: fcm_xy, ürün uzayında kümeleme yaptığı için kuralları y'nin hızlı değiştiği "
  "merkez bölgeye yoğunlaştırmakta, uçlardaki düşük genlikli dalgalanmaları ise yeterli kuralla kapsayamamaktadır. random, lhs ve fcm "
  "ile eğitilen modeller bu fold'da hedefle görsel olarak örtüşmektedir.")
rows = []
for mf in ("gauss", "trap"):
    for opt, olab in (("lm_tr", "LM-TR"), ("lm", "LM"), ("adam", "Adam")):
        r_ = [f"{mf.capitalize()} {olab}"]
        for i in ("random", "lhs", "fcm"):
            v = p2[(p2.variant == mf) & (p2.optimizer == opt) & (p2.init == i)].val_mse
            r_.append(f"{e(v.mean())} ± {e(v.std(ddof=0))}")
        rows.append(r_)
t5 = TAB("P2 — sinc, 5-katlı CV (her fold'da farklı başlatma), doğrulama MSE ortalama ± std.",
         ["Model / optimizer", "random (baseline)", "lhs (kontrol)", "fcm (önerilen)"], rows, [3.2, 4.4, 4.4, 4.4])
P(f"Çapraz doğrulamada (Tablo {t5}) fcm altı konfigürasyonun altısında hem ortalama hatayı hem varyansı düşürmektedir; trapezoid "
  "FN'de iyileşme 10–30 kattır. Gauss + LM-TR için fcm ile elde edilen 1,12×10⁻⁷ ± 8,4×10⁻⁸, makalenin raporladığı "
  "8,07×10⁻⁷ ± 5,78×10⁻⁷'den düşüktür. Beş eşleştirilmiş çiftte Wilcoxon testinin ulaşabileceği en küçük p değeri 0,0625 olduğundan "
  "bu tablo betimleyicidir; anlamlılık P1'de test edilmiştir.")
rows = []
for opt, olab in (("adam", "Adam"), ("lm", "LM"), ("lm_tr", "LM-TR")):
    r_ = [olab]
    for i in ("random", "lhs", "fcm", "fcm_xy"):
        v = p3[(p3.optimizer == opt) & (p3.init == i)].val_mse
        r_.append(e(v.mean()))
    rows.append(r_)
dr = p3[p3.optimizer == "lm_tr"].groupby("init")
rows.append(["Ölü kural, sonda (LM-TR)"] + [f"{dr.dead_rules_final.mean()[i]:.2f}" for i in ("random", "lhs", "fcm", "fcm_xy")])
rows.append(["Kapsanmayan örnek (LM-TR)"] + [f"%{100*dr.uncovered_val.mean()[i]:.0f}" for i in ("random", "lhs", "fcm", "fcm_xy")])
rows.append(["Kapsama oranı ρ (LM-TR)"] + [f"{dr.rho_mean.mean()[i]:.2f}" for i in ("random", "lhs", "fcm", "fcm_xy")])
t6 = TAB("P3/P4 — Trigonometrik veri, 10 eşleştirilmiş çift: ortalama doğrulama MSE ve yorumlanabilirlik ölçütleri.",
         ["", "random", "lhs", "fcm", "fcm_xy"], rows, [4.4, 3.0, 3.0, 3.0, 3.0])
P(f"Trigonometrik veride (Tablo {t6}) LM ile üç yöntem de anlamlı biçimde iyidir (fcm: 2,33×10⁻² → 6,20×10⁻³, p = 0,014). LM-TR ile "
  "lhs (p = 0,049) ve fcm_xy (p = 0,010) anlamlıdır ve makalenin 1,09×10⁻² ortalamasının altına inmektedir; ancak fcm'nin iyileşmesi "
  "anlamlı değildir (p = 0,49). Tablodaki açıklama şudur: fcm başlatmasında doğrulama örneklerinin %6'sını hiçbir kural "
  "kapsamamaktadır. Ölü kural sayısı tüm önerilen yöntemlerde sıfıra (ya da sıfıra yakın) inmektedir. lhs her durumda tutarlıdır ama "
  "kuralları genişletir (ρ = 0,70); fcm yerel kurallar üretir (ρ ≈ 0,50).")
fp3 = FIG(F / "p3_trig_box.png", "P3 — Trigonometrik veride başlatma yöntemine göre doğrulama hatasının dağılımı (10 çalıştırma).", 16.5)

r1 = p1[(p1.optimizer == "lm_tr")]
rb, rf = r1[r1.init == "random"], r1[r1.init == "fcm"]
q = p3[p3.optimizer == "lm"]
qb, qf = q[q.init == "random"], q[q.init == "fcm"]
tb_, tf_ = (rb.time_s + rb.init_time_s).mean(), (rf.time_s + rf.init_time_s).mean()
t7 = TAB("Baseline ve önerilen yöntemin özet karşılaştırması (yönerge §6.2 formatı).",
         ["Performans ölçütü", "Baseline (random)", "Önerilen (fcm)", "Değişim"],
         [["sinc LM-TR yakınsama oranı (P1)", f"{(rb.val_mse<SUCC).mean():.0%}", f"{(rf.val_mse<SUCC).mean():.0%}", "+55 puan (p < 0,001)"],
          ["sinc LM-TR medyan val. MSE (P1)", e(rb.val_mse.median()), e(rf.val_mse.median()), f"~{rb.val_mse.median()/rf.val_mse.median():,.0f}× düşük".replace(",", ".")],
          ["trig LM ort. val. MSE (P3)", e(qb.val_mse.mean()), e(qf.val_mse.mean()), f"{qb.val_mse.mean()/qf.val_mse.mean():.1f}× düşük (p = 0,014)".replace(".", ",", 1)],
          ["trig LM RMSE (P3)", f"{np.sqrt(qb.val_mse.mean()):.3f}", f"{np.sqrt(qf.val_mse.mean()):.3f}",
           f"{100*(np.sqrt(qf.val_mse.mean())-np.sqrt(qb.val_mse.mean()))/np.sqrt(qb.val_mse.mean()):+.0f}%"],
          ["sinc LM-TR toplam süre [s] (P5)", f"{tb_:.1f}", f"{tf_:.1f}", f"{100*(tf_-tb_)/tb_:+.0f}%"]],
         [5.4, 3.4, 3.4, 4.4])
H("8.3 Kural düzeyinde yorumlama örneği (P6)", 2)
p6 = pd.read_csv(T / "p6_rules.csv")
p6f = p6[p6.init == "fcm"]
t6r = TAB("P6 — FCM ile başlatılıp LM ile eğitilen [3,1,7] modelin kuralları (trigonometrik veri, fold 0). \"Veri\" sütunu, kuralın "
          "ateşlendiği doğrulama örneklerindeki gerçek y'nin ateşleme-ağırlıklı ortalamasıdır.",
          ["Kural", "EĞER", "İSE", "Önem", "Veri (y)", "Yön tutarlı"],
          [[r.rule, r["if"].replace("x1", "x₁").replace("x2", "x₂").replace("x3", "x₃"), r.then,
            f"{r.mean_significance:.2f}", f"{r.y_true_where_fires:.2f}", "evet" if r.direction_ok else "hayır"]
           for _, r in p6f.iterrows()], [1.3, 6.8, 2.0, 1.4, 1.6, 1.9])
pr = p6[p6.init == "random"]
P(f"Tablo {t6r}, FRBN'nin temel vaadi olan dilsel okunabilirliği önerilen başlatmayla eğitilmiş bir model üzerinde göstermektedir. "
  "Öncül etiketleri, trapezoid çekirdeğinin merkezine göre [0, 1] aralığının üçte birlik dilimlerinden (Düşük / Orta / Yüksek) "
  "verilmiştir; bir girişin üyeliği tüm aralıkta 1 ise kural o girişi yok sayar (\"herhangi\"). Örneğin \"EĞER x₂ Düşük VE x₃ Orta "
  "İSE y Düşük\" kuralı, sin(2πx₂) > 0 ve cos(2πx₃) < 0 olduğu bölgede çarpımın negatif olmasıyla tutarlıdır. \"Yön tutarlı\" sütunu, "
  "kuralın y'yi ortalamanın üstüne mi altına mı ittiğinin gerçek veriyle aynı yönde olup olmadığını gösterir. FCM ile başlatılan "
  f"modelde {int(p6f.direction_ok.sum())}/7 kural yön olarak tutarlıdır ve ölü kural yoktur (doğrulama MSE "
  f"{e(p6f.val_mse.iloc[0])}). Aynı tohumla rastgele başlatılan modelde {int(pr.direction_ok.sum())}/7 kural tutarlıdır, "
  f"{int(pr.dead.sum())} kural ölüdür ve doğrulama MSE {e(pr.val_mse.iloc[0])}'dir.")
P(f"Bu tablo tek bir modelden alınmış açıklayıcı bir örnektir, istatistiksel bir karşılaştırma değildir. Ayrıca her iki modelde de "
  f"sonuç kümelerinin merkezleri çıkış aralığının dışına taşmaktadır (fcm: {p6f.cons_center.min():.1f} ile {p6f.cons_center.max():.1f}; "
  f"random: {pr.cons_center.min():.1f} ile {pr.cons_center.max():.1f}). Kısıtsız eğitim, örtüşen kuralların katkılarını dengelemek için "
  "sonuçları aralık dışına itmektedir; bu durum sonuç etiketlerinin ancak yön olarak yorumlanabileceği anlamına gelir. Etiketler bu "
  "yüzden aralığa kırpılarak verilmiştir. Sonuç kümelerinin çıkış aralığıyla sınırlandırılması yorumlanabilirliği artıracak bir kısıttır.")

P(f"**Maliyet (P5).** FCM başlatması sinc'te yaklaşık 2,1 s, trigonometrik veride 0,43 s sürmekte ve toplam süreyi %19–27 "
  "artırmaktadır; önceden öngördüğümüz %5 sınırı aşılmıştır. lhs'in maliyeti ise ihmal edilebilir düzeydedir (~0,003 s). Öte yandan "
  "yakınsama oranı 9/20'den 20/20'ye çıktığı için başarılı bir model elde etmenin beklenen maliyeti yaklaşık yarıya inmektedir. "
  f"Tablo {t7} başlıca sonuçları özetlemektedir.")

# ================================================================= 9
H("9. Tartışma (Discussion)")
P("**Başlatma neden bu kadar önemli?** FRBN'de bir kuralın parametreleri yalnızca o kuralın ateşlendiği örneklerden gradyan alır. "
  "Sinir ağlarında her nöron her örnekte bir miktar etkili olduğu için kör bölgeler oluşmaz; FRBN'de ise kuralların yerelliği, "
  "yorumlanabilirliği sağlayan özellik olmakla birlikte optimizasyonu kırılgan hale getirir. LM yerel bir yöntem olduğu için bu "
  "etki LM ailesinde en büyüktür. Bu bulgu, makalenin \"NN eğitimi daha kararlıdır\" gözlemini açıklamakta ve güçlendirmektedir.")
P("**Kazancın kaynağı kapsamadır.** Kümeleme yapmayan lhs kontrolü birçok konfigürasyonda fcm kadar, bazen ondan iyidir. Bu, "
  "iyileşmenin çoğunun \"her bölgede en az bir kural olması\"ndan geldiğini; FCM'nin veri yapısını öğrenmesinin bu düşük boyutlu "
  f"problemlerde ek bir avantaj sağlamadığını göstermektedir. Bu sonuç, FCM'nin ızgara başlatmadan özellikle yüksek boyutta daha iyi "
  f"olduğunu bulan Shi vd. {cite('fcmrdpa')} ile çelişmez; iki çalışmanın boyut aralıkları farklıdır. Katkı, en doğru biçimde "
  "\"veri güdümlü, kapsama-bilinçli başlatma\" olarak tanımlanabilir; lhs ve fcm bunun iki uygulamasıdır.")
P(f"**FCM'nin zayıf noktası.** Çok boyutlu veride FCM'nin bulanık standart sapmasından gelen genişlikler kapsamayı garanti etmemekte "
  "(trigonometrik veride %6 kapsanmayan örnek) ve LM-TR + fcm'nin anlamsız sonucunu açıklamaktadır. Yani katkının çözdüğü problem, "
  "belirli koşullarda önerilen yöntemin kendisinde de ortaya çıkabilmektedir. Alt genişlik sınırının tam hücre boyuna çıkarılması ya "
  "da kapsanmayan örneklere yeni kural eklenmesi olası düzeltmelerdir. Bu düzeltmeler, sonuçlar görüldükten sonra yapılırsa sonuçlara "
  "göre ayar yapılmış olacağından uygulanmamış, gelecek çalışma olarak bırakılmıştır.")
P(f"**Yorumlanabilirlik.** Önerilen başlatma ölü kuralları ortadan kaldırarak her kuralın modelde bir işlevi olmasını sağlar. lhs ise "
  f"kuralları globalleştirerek (ρ = 0,70) yerel dilsel anlamı zayıflatır; bu bir doğruluk–yorumlanabilirlik dengesidir. Makalenin "
  f"gelecek çalışma olarak önerdiği Ruspini bölüntüsü {cite('ruspini')} bu dengeyi kurmanın bir yolu olabilir.")

# ================================================================= 10
H("10. Sınırlılıklar (Limitations)")
B(["Deneyler 1 ve 3 boyutlu sentetik verilerle sınırlıdır; yüksek boyutlu gerçek veri (ör. makaledeki MEMS) test edilmemiştir.",
   "Bakteriyel memetik algoritma (BMA) kapsam dışında bırakılmıştır; karşılaştırmalar Adam, LM ve LM-TR ile yapılmıştır.",
   "Yalnızca tek katmanlı FRBN'ler denenmiştir; çok katmanlı ağlarda yalnızca ilk katman veriden başlatılabilmektedir.",
   "P1 tek bir veri bölmesinde yapılmıştır; P2'de 5 çift olduğu için istatistiksel güç düşüktür. P6'daki kural analizi tek bir modele "
   "dayanan açıklayıcı bir örnektir.",
   "Kural parametreleri kısıtsız eğitildiği için sonuç kümeleri çıkış aralığının dışına taşabilmekte ve kuralların dilsel yorumu "
   "zayıflamaktadır; Ruspini bölüntüsü ya da aralık kısıtları denenmemiştir.",
   "y normalizasyonu makalede belirtilmediğinden trigonometrik sonuçların makaleyle mutlak karşılaştırması temkinli yorumlanmalıdır.",
   "Süre ölçümleri tek bir dizüstü bilgisayarda, CPU üzerinde yapılmıştır."])

# ================================================================= 11
H("11. Sonuç (Conclusion)")
P("FRBN bağımsız olarak yeniden gerçekleştirilmiş; sinc'te makaledeki 10⁻⁷–10⁻⁸ seviyesine ulaşılabildiği, sinir ağının daha kararlı "
  "eğitildiği ve trapezoid modelin Gauss modelden kötü olduğu doğrulanmıştır. Trigonometrik veride makalenin optimizer sıralaması "
  "tekrarlanamamıştır. En önemli bulgu, FRBN'nin LM ile eğitiminin başlatmaya çok duyarlı olması ve bunun nedeninin rastgele "
  "başlatmanın kuralları veri üzerine dengesiz dağıtması, ateşlenmeyen kurallarda ise gradyanın sıfır olmasıdır. Önerilen kapsama-bilinçli başlatma, mimariyi değiştirmeden yakınsamayı 9/20'den 20/20'ye "
  "çıkarmış, sinc'te tüm konfigürasyonlarda hatayı ve varyansı düşürmüş ve ölü kuralları ortadan kaldırmıştır. Gelecek çalışma olarak "
  "kapsamayı garanti eden genişlik seçimi, yüksek boyutlu gerçek veri setleri ve çok katmanlı ağlarda katman katman başlatma "
  "önerilmektedir.")

# ================================================================= AI disclosure
H("Üretken Yapay Zekâ Kullanım Beyanı (Generative AI Usage Disclosure)")
P("Bu projede Anthropic'in Claude modeli (Claude Code aracılığıyla) aşağıdaki işlerde yoğun olarak kullanılmıştır: kodun yazılması "
  "ve hata ayıklama, deney betiklerinin ve istatistik analizlerinin hazırlanması, proje belgelerinin ve bu raporun taslağının "
  "yazılması, makalenin ve resmi kodun incelenmesi ile literatür kaynaklarının bulunması. Tüm akademik kaynakların künyeleri DOI "
  "kayıtlarından (Crossref), arXiv'den veya yayıncı sayfasından doğrulanmış; doğrulanamayan bir aday kaynak kullanılmamıştır. "
  "Raporlanan bütün sayılar repodaki deney betiklerinin ürettiği sonuç dosyalarından alınmıştır ve tekrar üretilebilir.")
P("Yapay zekâ çıktıları doğrudan kabul edilmemiş, öğrenci tarafından incelenmiş ve doğrulanmıştır. Kodun tamamı satır satır "
  "incelenmiş; her modülün (üyelik fonksiyonları, bulanık katman, Center of Sums durulaştırması, Levenberg–Marquardt optimizer'ı, "
  "başlatma yöntemleri) görevi ve makaledeki karşılık gelen denklemi kontrol edilmiştir. Birim testleri ve deney betikleri öğrencinin "
  "bilgisayarında çalıştırılmış, rapordaki tablo ve şekiller bu çalıştırmaların ürettiği sonuç dosyalarıyla ve referans makalenin "
  "tablolarıyla karşılaştırılmıştır. Projenin kapsamı (MEMS veri setinin ve bakteriyel memetik algoritmanın kapsam dışı bırakılması), "
  "katkının seçimi, raporun dili ve teslim edilen bütün dosyalar öğrenci tarafından belirlenmiş ve onaylanmıştır. Süreç boyunca "
  "yapay zekâ çıktılarında bulunan hatalar düzeltilmiştir: ilgisiz bir makaleye çıkan bir DOI kaynakçadan çıkarılmış, sinc deneyindeki "
  "başarısızlık için ilk ileri sürülen açıklama (\"kaçırılan bölgede hiçbir kural ateşlenmiyor\") ölçümle yanlışlanmış ve Bölüm 8.1'deki "
  "biçimiyle düzeltilmiştir. Öğrenci, sunulan kodu, yöntemleri ve sonuçları açıklayabilecek düzeyde anladığını beyan eder.")

# ================================================================= references
H("Kaynaklar (References)")
for i, k in enumerate(ORDER, start=1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.first_line_indent = Cm(-0.8)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f"[{i}] {REFS[k]}"); r.font.size = Pt(9)

unused = [k for k in REFS if k not in ORDER]
doc.core_properties.author = "Turan ÇALDIRAN"
doc.core_properties.title = "FRBN: Yeniden Gerçekleştirme ve Kapsama-Bilinçli Kural Başlatma"
# mark the file as a Word 2013+ document so Word does not open it in "Compatibility Mode"
settings = doc.settings.element
compat = settings.find(qn("w:compat"))
if compat is None:
    compat = OxmlElement("w:compat"); settings.append(compat)
for old in compat.findall(qn("w:compatSetting")):
    if old.get(qn("w:name")) == "compatibilityMode":
        compat.remove(old)
cs = OxmlElement("w:compatSetting")
cs.set(qn("w:name"), "compatibilityMode"); cs.set(qn("w:uri"), "http://schemas.microsoft.com/office/word"); cs.set(qn("w:val"), "15")
compat.append(cs)
doc.save(OUT)
print(f"saved {OUT}  figures={fig_no[0]} tables={tab_no[0]} refs={len(ORDER)} unused={unused}")
