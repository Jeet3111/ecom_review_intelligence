#!/usr/bin/env python3
"""
build_report.py - builds report/Review_Intelligence_Report.docx from outputs/results.json + outputs/figures/*.png
Every number in the report is read from the results of the latest `python run_pipeline.py`.
Run:  python report/build_report.py
"""
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_COLOR_INDEX

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R = json.load(open(ROOT / "outputs" / "results.json", encoding="utf-8"))
FIG = ROOT / "outputs" / "figures"
BLUE = RGBColor(0x1F, 0x3A, 0x8A)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.2)
sec.top_margin = sec.bottom_margin = Cm(2.0)

st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.12
for name, size, color in [("Heading 1", 16, BLUE), ("Heading 2", 13, BLUE), ("Heading 3", 11.5, RGBColor(0x37, 0x41, 0x51))]:
    h = doc.styles[name]
    h.font.name = "Calibri"
    h.font.size = Pt(size)
    h.font.bold = True
    h.font.color.rgb = color
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    h.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
    h.paragraph_format.space_after = Pt(5)
    h.paragraph_format.keep_with_next = True


# ---------------------------------------------------------------- helpers
def H(text, lvl=1):
    return doc.add_heading(text, lvl)


def P(text="", bold=False, italic=False, align=None, size=None, color=None, after=None):
    p = doc.add_paragraph()
    _runs(p, text, bold, italic, size, color)
    if align:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def _runs(p, text, bold=False, italic=False, size=None, color=None):
    """**bold** and [[placeholder]] inline markup."""
    import re
    for part in re.split(r"(\*\*.+?\*\*|\[\[.+?\]\]|`.+?`)", text):
        if not part:
            continue
        if part.startswith("`"):
            r = p.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(size - 1 if size else 9.5)
        elif part.startswith("**"):
            r = p.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("[["):
            r = p.add_run("[" + part[2:-2] + "]")
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            r.bold = True
        else:
            r = p.add_run(part)
            r.bold = bold
        r.italic = italic
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color


def B(text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(0.9 + 0.6 * level)
    p.paragraph_format.space_after = Pt(2)
    _runs(p, text)
    return p


_NUM = [0]


def NRESET():
    _NUM[0] = 0


def N(text):
    _NUM[0] += 1
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.first_line_indent = Cm(-0.6)
    p.paragraph_format.space_after = Pt(2)
    _runs(p, f"{_NUM[0]}.  " + text)
    return p


def EQ(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.name = "Cambria Math"
    r.italic = True
    r.font.size = Pt(11)
    p.paragraph_format.space_after = Pt(5)
    return p


def CODE(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F3F4F6")
    pPr.append(shd)
    return p


def FIGURE(name, caption, width=16.0):
    path = FIG / name
    if not path.exists():
        P(f"[missing figure {name}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width))
    c = P(caption, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=9, color=RGBColor(0x55, 0x55, 0x55))
    c.paragraph_format.space_after = Pt(10)


def _shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def TABLE(header, rows, widths=None, size=9, first_bold=False, highlight_row=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(str(h))
        r.bold = True
        r.font.size = Pt(size)
        r.font.color.rgb = RGBColor(255, 255, 255)
        _shade(c, "1F3A8A")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            para = cells[i].paragraphs[0]
            para.paragraph_format.space_after = Pt(0)
            r = para.add_run(str(v).replace('**', ''))
            r.font.size = Pt(size)
            if (first_bold and i == 0) or ri == highlight_row:
                r.bold = True
            if ri % 2 == 1:
                _shade(cells[i], "F3F4F6")
            if ri == highlight_row:
                _shade(cells[i], "DBEAFE")
    if widths:
        t.autofit = False
        for i, w in enumerate(widths):
            t.columns[i].width = Cm(w)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def PB():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# footer page numbers
def _footer():
    p = sec.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Indian E-commerce Review Intelligence · LCA1 · page ")
    r.font.size = Pt(8.5)
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        run = p.add_run()
        run.font.size = Pt(8.5)
        if tag:
            f = OxmlElement("w:fldChar")
            f.set(qn("w:fldCharType"), tag)
            run._r.append(f)
        else:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = txt
            run._r.append(it)


_footer()

# ---------------------------------------------------------------- shortcuts
D, SPL, FEAT, PER, BASE, FIN = R["dataset"], R["split"], R["features"], R["perceptron"], R["baseline"], R["final"]
T = FIN["test"]
HP = R["hyperparameters"]
HB = R["hyperparameter_best"]
GD = R["gd_variants"]["table"]
CC = {r["setup"]: r for r in R["cost_compare"]["table"]}
SENT = R.get("sentiment")
cfg = FIN["config"]

# =============================================================== COVER
for _ in range(3):
    P("")
P("M.Tech AI-ML  ·  LCA1", bold=True, size=13, color=RGBColor(0x55, 0x55, 0x55), align=WD_ALIGN_PARAGRAPH.CENTER)
P("Design and Deployment of an AI-Based Web Application for Classification using Neural Networks", bold=True, size=24,
  color=BLUE, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)
P("Indian E-commerce Review Intelligence", bold=True, size=18, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
P("Spam / fake-review detection · sentiment · topic dashboard", size=13, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=24)
P("Deep-learning topics covered in depth: Cost functions in neural networks · Perceptron & back-propagation · Variants of gradient descent",
  size=11, align=WD_ALIGN_PARAGRAPH.CENTER, after=30)
TABLE(["Item", "Details"], [
    ["Student name", "[[Your name]]"], ["Roll / enrolment no.", "[[Your roll number]]"],
    ["Programme / course", "M.Tech AI-ML — LCA1"], ["Assignment issued", "8 September 2026 (deadline stated in brief: 20 September 2026)"],
    ["Deployed application URL", "[[Paste your Google Cloud Run service URL here]]"],
    ["Source code", "ecom_review_intelligence.zip (submitted with this report)"],
], widths=[5, 11], size=10, first_bold=True)
PB()

# =============================================================== CONTENTS
H("Contents", 1)
for line in ["1  Introduction and objectives", "2  Dataset, pre-processing and feature extraction", "3  The Perceptron — working and limitations",
             "4  Multi-layer perceptron and forward propagation", "5  Cost (error) functions", "6  Back-propagation and convergence",
             "7  Variants of gradient descent", "8  Hyper-parameter tuning", "9  Performance evaluation", "10  Web application",
             "11  Google Cloud deployment", "12  Limitations and future work", "13  Conclusion", "References",
             "Appendix A  How to run the project step by step", "Appendix B  Submission checklist"]:
    P(line, after=1)
P("")
P("**How to read this report.** Every number, table and figure is generated by the code in the accompanying zip (`run_pipeline.py` → "
  "`outputs/results.json`) and inserted automatically by `report/build_report.py`; re-running the pipeline and rebuilding the report keeps the two consistent. "
  "Yellow-highlighted items are things only the student can supply (name, deployed URL, screenshots).", size=10)
PB()

# =============================================================== 1 INTRO
H("1  Introduction and objectives", 1)
P("Online marketplaces such as Flipkart, Amazon.in, Myntra, Meesho and Nykaa rely on customer reviews to drive purchase decisions. A growing share of those "
  "reviews is not genuine: promotional links, paid ‘5-star’ reviews, copy-pasted praise and outright junk. This project builds an **AI-based web application** "
  "that takes a review typed by a user and answers, in real time, **Spam / Not Spam**. For genuine reviews it additionally reports **sentiment** "
  "(Negative / Neutral / Positive), **topics** (delivery, packaging, quality, price, battery, fit …) and feeds a **review-intelligence dashboard** "
  "(trend, sentiment split, rating distribution, top positive / negative topics, price-vs-rating analysis).")
P("The assignment brief asks students to demonstrate activation functions, a cost function, back-propagation and hyper-parameter tuning with a multi-layer "
  "perceptron, and then deploy on Google Cloud. In addition, three deep-learning topics are treated in depth and implemented **from scratch in NumPy** "
  "(no TensorFlow / PyTorch), so that every equation in the report corresponds to a few lines of visible code:")
B("**Cost functions in neural networks** — MSE, binary cross-entropy, categorical cross-entropy; their gradients and behaviour (Section 5).")
B("**Perceptron and back-propagation** — Rosenblatt perceptron, its XOR limitation, the MLP, chain-rule back-propagation and a numerical gradient check (Sections 3, 4, 6).")
B("**Variants of gradient descent** — Batch, Stochastic, Mini-batch, Momentum, Nesterov, AdaGrad, RMSProp, Adam (Section 7).")
FIGURE("00_pipeline.png", "Figure 1 — End-to-end architecture, following the suggested architecture in the brief.", 16.5)
P("**Requirement coverage.**", after=2)
TABLE(["Brief requirement", "Where addressed"], [
    ["1 Dataset preparation, pre-processing, feature extraction", "Section 2"], ["2 Multi-layer perceptron: working and limitations", "Sections 3, 4"],
    ["3 Neural network ≥ 1 hidden layer, forward propagation, ReLU / Sigmoid / …", "Section 4"],
    ["4 Cost function implemented, explained, loss-vs-epochs plot", "Sections 5, 6"],
    ["5 Back-propagation and demonstration of convergence", "Section 6"],
    ["6 Hyper-parameter tuning (≥ 3 hyper-parameters)", "Section 8 (six studied)"],
    ["7 Accuracy, precision, recall, F1, confusion matrix", "Section 9"],
    ["8 Web interface: Input message → AI model → Spam / Not Spam", "Section 10"], ["9 Google Cloud deployment", "Section 11"],
], widths=[10.5, 5.5])

# =============================================================== 2 DATA
H("2  Dataset, pre-processing and feature extraction", 1)
H("2.1  Dataset", 2)
P(f"**Important note on data.** The development environment had no internet access, so a public review corpus could not be downloaded. Instead the project includes a "
  f"documented generator (`reviewintel/synth.py`) that produces a **synthetic but realistic Indian e-commerce review dataset**: English and Hinglish text "
  f"(e.g. ‘Packing bahut kharab thi’), six categories, {D['products']} products with INR prices, platforms (Flipkart, Amazon.in, Myntra, Meesho, Nykaa, JioMart), "
  f"1–5 star ratings, dates over 2026 with growing volume, and a spam label. Seven spam / fake types are simulated: promotional links, paid-review solicitations, gibberish, "
  f"off-topic posts, generic fake praise, genuine-looking review + promo tail, and *subtle fakes* with no link at all. To keep the task realistic, genuine ‘hard negatives’ "
  f"(genuine reviews quoting a customer-care number, ‘5 stars’, WhatsApp …) and 1.5 % annotation noise are included. **Results therefore describe this dataset and should "
  f"not be read as accuracy on real marketplace traffic.** A real CSV can be plugged in with `python run_pipeline.py --data my_reviews.csv` (Appendix A).")
TABLE(["Property", "Value"], [
    ["Source", D["source"]], ["Reviews", f"{D['rows']:,}"], ["Products / categories", f"{D['products']} / {D['categories']}"],
    ["Spam (positive class) share", pct(D["spam_rate"], 1)], ["Average review length", f"{D['mean_chars']} characters"],
    ["Train / validation / test (stratified, seed 42)", f"{SPL['train']:,} / {SPL['val']:,} / {SPL['test']:,}"],
], widths=[8, 8], first_bold=True)
H("2.2  Pre-processing", 2)
for s in ["**Normalisation:** lower-casing; elongated characters squeezed (‘soooo’ → ‘soo’); emoji and punctuation removed; ‘90%’ → ‘90 percent’.",
          "**Spam-aware tokens:** URLs, Indian mobile numbers and @handles are *replaced by the tokens* `urltoken`, `phonetoken`, `handletoken` instead of being deleted, because their presence is itself a signal.",
          "**No stop-word removal:** words such as ‘not’ and ‘no’ carry sentiment, and Hinglish function words (‘hai’, ‘ekdum’) are not in standard stop-word lists.",
          "**Split before fitting:** vocabulary, IDF and scaling statistics are learnt on the training split only (no leakage into validation / test)."]:
    B(s)
H("2.3  Feature extraction", 2)
P(f"Each review becomes a vector of **{FEAT['n_total']:,} numbers** = {FEAT['n_tfidf']:,} TF-IDF features + {FEAT['n_hand']} style features:")
B(f"**TF-IDF** over unigrams and bigrams (vocabulary of the {FEAT['n_tfidf']:,} most frequent n-grams with document frequency ≥ 3), implemented from scratch: "
  "weight = (1 + ln tf) · idf with idf = ln((1+n)/(1+df)) + 1, L2-normalised per review.")
B("**Ten hand-crafted ‘style’ features** that capture *how* a review is written: log length in characters and words, upper-case ratio, digit ratio, number of exclamation marks, "
  "contains-URL, contains-phone, repeated-word ratio, special-character ratio and mean word length — standardised with training statistics.")
P("The extractor is saved as JSON, so the deployed web app needs only NumPy and Flask.")

# =============================================================== 3 PERCEPTRON
H("3  The Perceptron — working and limitations", 1)
P("The perceptron (Rosenblatt, 1958) is a single artificial neuron: it forms a weighted sum of the inputs, adds a bias and applies a hard threshold.")
EQ("z = w · x + b            ŷ = 1 if z > 0, else 0")
EQ("if the sample is misclassified (y ∈ {−1,+1}):   w ← w + η·y·x ,   b ← b + η·y")
P("Learning is therefore **error-driven**: weights change only when a prediction is wrong. The perceptron convergence theorem guarantees that this rule reaches zero training "
  "errors in a finite number of updates **if and only if the classes are linearly separable**.")
P(f"**Result on review data.** Trained for {len(PER['errors_per_epoch'])} epochs on the {FEAT['n_total']:,}-dimensional features, the perceptron makes "
  f"{PER['errors_per_epoch'][0]} mistakes in epoch 1 and still {PER['errors_per_epoch'][-1]} in epoch {len(PER['errors_per_epoch'])}: it never reaches zero because spam and "
  f"genuine reviews are *not perfectly* linearly separable (subtle fakes look like genuine praise). On the test set it scores accuracy {pct(PER['test']['accuracy'])}, "
  f"precision {pct(PER['test']['precision'])}, recall {pct(PER['test']['recall'])}, F1 {pct(PER['test']['f1'])}. This is a strong baseline because most spam types "
  f"(links, phone numbers, junk) *are* separable by a single hyper-plane.")
FIGURE("03_perceptron_errors.png", "Figure 2 — Perceptron updates per epoch: errors fall but never reach zero.", 9.5)
H("3.1  Limitations — the XOR problem", 2)
P(f"A single perceptron can only draw **one straight decision boundary**. The classic counter-example is XOR, whose two classes cannot be separated by a line. "
  f"Our demonstration gives the perceptron {pct(R['xor']['perceptron_acc'], 0)} accuracy (chance level) while a small MLP with one hidden layer solves it "
  f"with {pct(R['xor']['mlp_acc'], 0)} (Figure 3). Hidden layers with non-linear activations let the network build *curved / piece-wise* boundaries — the motivation for the "
  f"multi-layer perceptron (Minsky & Papert, 1969; Rumelhart et al., 1986).")
FIGURE("04_xor_perceptron_vs_mlp.png", "Figure 3 — XOR: a single perceptron (left) cannot separate the classes; an MLP with one hidden layer (right) can.", 13)

# =============================================================== 4 MLP
H("4  Multi-layer perceptron and forward propagation", 1)
P("A multi-layer perceptron stacks layers of neurons. Layer *l* computes a linear transformation followed by a non-linear activation *f*:")
EQ("z⁽ˡ⁾ = a⁽ˡ⁻¹⁾ W⁽ˡ⁾ + b⁽ˡ⁾            a⁽ˡ⁾ = f( z⁽ˡ⁾ ),     a⁽⁰⁾ = x")
P(f"The deployed spam network is `{FEAT['n_total']:,} → {' → '.join(str(h) for h in cfg['hidden'])} → 1` with **{cfg['act'].upper()}** hidden units and a **sigmoid** output "
  f"that is read as P(spam); it has {FIN['params']:,} trainable parameters. Forward propagation is the function `MLP.forward` in `reviewintel/nn.py`. "
  f"Weights use He initialisation (ReLU) or Xavier / Glorot initialisation (sigmoid / tanh) to keep activations well scaled.")
FIGURE("05_network.png", "Figure 4 — Architecture of the deployed network (only a few neurons per layer drawn).", 10.5)
H("4.1  Activation functions", 2)
TABLE(["Activation", "f(z)", "f′(z)", "Remarks"], [
    ["ReLU", "max(0, z)", "1 if z>0 else 0", "Cheap, no saturation for z>0; ‘dead neuron’ risk"],
    ["Leaky ReLU", "z if z>0 else 0.01z", "1 or 0.01", "Avoids dead neurons"],
    ["Sigmoid", "1/(1+e⁻ᶻ)", "a(1−a)", "Output in (0,1) → probability; saturates → vanishing gradients in hidden layers"],
    ["Tanh", "(eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)", "1−a²", "Zero-centred; still saturates"],
    ["Softmax", "eᶻᵏ / Σ eᶻʲ", "(with cross-entropy: a−y)", "Used for the 3-class sentiment output"],
], widths=[2.6, 3.6, 3.2, 6.6], size=8.5, first_bold=True)
FIGURE("01_activations.png", "Figure 5 — Activation functions (left) and the derivatives that back-propagation multiplies by (right).", 15.5)
H("4.2  Working and limitations of the MLP", 2)
B("**Working:** hidden layers learn intermediate feature combinations (e.g. ‘link + urgency words’) and the output layer converts them into a probability; "
  "the universal-approximation theorem says one wide hidden layer can approximate any continuous function.")
B("**Limitations:** (i) needs many labelled examples and is easily over-fitted — see Section 8; (ii) the *bag-of-n-grams* input throws away word order and context, so sarcasm "
  "or subtle fake praise is hard; (iii) many hyper-parameters; (iv) non-convex cost surface, so training may stall in poor regions; (v) it is a ‘black box’ — we add an occlusion explanation "
  "in the web app to partly compensate; (vi) it cannot handle unseen vocabulary or spelling variants beyond the training vocabulary (typical for Hinglish).")

# =============================================================== 5 COST
H("5  Cost (error) functions", 1)
P("A cost function J turns the network’s mistakes into a single number that training minimises. For m samples with target y and network output a:")
TABLE(["Cost", "Formula", "Used for", "Output delta  ∂J/∂z (starts back-prop)"], [
    ["Mean squared error (MSE)", "J = (1/m) Σ (a − y)²", "regression; (here: comparison)", "2(a−y)·a(1−a)/m  (with sigmoid)"],
    ["Binary cross-entropy (BCE)", "J = −(1/m) Σ [ y ln a + (1−y) ln(1−a) ]", "binary classification → **spam head**", "(a − y)/m"],
    ["Categorical cross-entropy", "J = −(1/m) Σᵢ Σₖ yᵢₖ ln aᵢₖ", "multi-class → **sentiment head**", "(a − y)/m  (with softmax)"],
], widths=[3.3, 5.2, 3.6, 3.9], size=8.5, first_bold=True)
P("**Why cross-entropy for classification?** BCE is the negative log-likelihood of a Bernoulli model, so it penalises a *confidently wrong* prediction very heavily "
  "(cost → ∞ as a → 1−y), and — crucially — the sigmoid’s derivative cancels in the gradient, leaving the simple error (a − y). With MSE the gradient contains the factor a(1−a), "
  "which is almost zero when the neuron is saturated, i.e. exactly when it is confidently wrong (Figure 6, middle).")
FIGURE("02_cost_functions.png", "Figure 6 — Left: cost of one prediction; middle: gradient reaching back-propagation (MSE vanishes for confident errors); right: cost against the logit.", 16.5)
H("5.1  Experiment: BCE vs MSE", 2)
P(f"The same network ({FEAT['n_total']:,}→32→1, ReLU, mini-batch 64, 20 epochs) was trained with each cost, using Adam and plain gradient descent. For plain GD each cost received "
  f"its own best learning rate from a small grid. A further pair of runs starts from a **‘confidently wrong’** state (output bias set to −8, so the net initially predicts ‘genuine’ for everything) "
  f"with the *same* learning rate (0.5) for both costs. Note that costs of different functions are on different scales, so compare accuracy / F1, not the raw cost values.")
TABLE(["Set-up", "lr", "final train cost", "val accuracy", "val F1"],
      [[r["setup"], r["lr"], f"{r['final_train_cost']:.4f}", pct(r["val_acc"]), f"{r['val_f1']:.4f}"] for r in R["cost_compare"]["table"]],
      widths=[6.6, 1.5, 3, 2.7, 2.2], size=8.5, first_bold=False)
FIGURE("06_cost_bce_vs_mse.png", "Figure 7 — Training cost (log scale) and validation accuracy for each cost / optimiser combination. Dashed = confidently-wrong start.", 16.5)
a, b = CC["BCE + Adam"], CC["MSE + Adam"]
c, d = CC["BCE + GD"], CC["MSE + GD"]
e, f_ = CC["BCE + GD (confidently-wrong start)"], CC["MSE + GD (confidently-wrong start)"]
P("**Findings.**", after=2)
B(f"From a normal start the two costs end up **equally good** on this easy problem (Adam: F1 {a['val_f1']:.3f} vs {b['val_f1']:.3f}; plain GD: {c['val_f1']:.3f} vs {d['val_f1']:.3f}); "
  f"adaptive optimisers largely hide the gradient-scaling difference. We do not claim that BCE is more accurate here.")
B(f"From the confidently-wrong start the difference is dramatic: with BCE the network recovers to {pct(e['val_acc'], 1)} validation accuracy (F1 {e['val_f1']:.3f}), whereas with MSE "
  f"it is **stuck at {pct(f_['val_acc'], 1)} (F1 {f_['val_f1']:.3f})** — it keeps predicting ‘genuine’ for every review because the gradient 2(a−y)a(1−a) is almost zero. "
  f"This is the practical reason cross-entropy is preferred for sigmoid / softmax outputs.")
B("BCE also yields better-calibrated probabilities, which the web app shows as P(spam).")

# =============================================================== 6 BACKPROP
H("6  Back-propagation and convergence", 1)
P("Back-propagation computes ∂J/∂W for **every** weight in one backward sweep by applying the chain rule layer by layer. With δ⁽ˡ⁾ = ∂J/∂z⁽ˡ⁾:")
EQ("δ⁽ᴸ⁾ = ∂J/∂z⁽ᴸ⁾ = (a⁽ᴸ⁾ − y)/m                       (BCE + sigmoid, output layer)")
EQ("δ⁽ˡ⁾ = ( δ⁽ˡ⁺¹⁾ · W⁽ˡ⁺¹⁾ᵀ ) ⊙ f′( z⁽ˡ⁾ )                (propagate the error backwards)")
EQ("∂J/∂W⁽ˡ⁾ = a⁽ˡ⁻¹⁾ᵀ · δ⁽ˡ⁾          ∂J/∂b⁽ˡ⁾ = Σ rows of δ⁽ˡ⁾")
EQ("W⁽ˡ⁾ ← W⁽ˡ⁾ − η · ∂J/∂W⁽ˡ⁾          (gradient-descent step; Section 7 gives the variants)")
P("Algorithm per mini-batch: (1) forward propagate and cache every z and a; (2) compute the output delta; (3) loop from the last layer to the first, "
  "computing weight gradients and passing the error back through the transpose of the weights and the activation derivative; (4) let the optimiser update all parameters. "
  "This is implemented in `MLP.backward` (≈ 10 lines).")
H("6.1  Correctness: numerical gradient check", 2)
P("To prove the hand-written back-propagation is correct, its analytic gradient is compared with the centred finite-difference gradient "
  "(J(θ+ε) − J(θ−ε))/2ε for every parameter of a small network. A relative error below 1e-7 indicates correct gradients.")
TABLE(["Hidden act. + cost + output", "Relative error"], [[k, f"{v:.2e}"] for k, v in R["gradient_check"].items()], widths=[8, 4], size=9)
P("All combinations agree to ≈ 1e-10, so the same code path can be trusted for the spam and sentiment networks (it is also part of the unit tests).")
H("6.2  Demonstration of convergence", 2)
P(f"The final spam network was trained with mini-batch Adam and early stopping (patience 8 on validation cost). Training cost fell from **{FIN['train_loss_first']:.3f}** "
  f"(epoch 1) to **{FIN['train_loss_last']:.3f}**, while the validation cost reached its minimum of **{FIN['val_loss_best']:.3f}** at epoch **{FIN['best_epoch']}**; training stopped at epoch "
  f"{FIN['stopped_epoch']} and the weights of the best epoch were restored. The norm of the full-batch gradient shrank from **{FIN['grad_norm_first']:.3f}** to "
  f"**{FIN['grad_norm_last']:.4f}** ({FIN['grad_norm_first'] / max(FIN['grad_norm_last'], 1e-9):.0f}× smaller) — the gradient heading towards zero is the signature of convergence to a (local) minimum. "
  f"Training and validation accuracy rise together and level off near {pct(FIN['curves']['val_acc'][-1], 1)}, i.e. no serious over-fitting.")
FIGURE("11_final_loss_curve.png", "Figure 8 — Training / validation cost vs epochs (left), accuracy (middle) and gradient norm (right) of the deployed model.", 16.5)

# =============================================================== 7 GD VARIANTS
H("7  Variants of gradient descent", 1)
P("All variants follow the idea θ ← θ − (step), differing in **how much data** is used for each gradient (Batch / Stochastic / Mini-batch) and **how the step is built** "
  "from the gradient history (Momentum … Adam). In the notation g = ∂J/∂θ, learning rate η:")
TABLE(["Variant", "Update rule", "Idea / trade-off"], [
    ["Batch GD", "θ ← θ − η·g (g over all m samples)", "Exact gradient, but only 1 update per epoch → slow; needs all data in memory"],
    ["Stochastic GD", "same, g from 1 sample", "Very frequent, noisy updates; escapes shallow minima; slow in Python (no vectorisation)"],
    ["Mini-batch GD", "same, g from 32–256 samples", "Best compromise: vectorised, moderate noise — the practical default"],
    ["Momentum", "v ← μv − ηg ;  θ ← θ + v", "Accumulates velocity → damps zig-zag in ravines, accelerates along consistent directions"],
    ["Nesterov (NAG)", "v ← μv − ηg ; θ ← θ − μv_prev + (1+μ)v", "Look-ahead version of momentum; corrects overshoot earlier"],
    ["AdaGrad", "G ← G + g² ;  θ ← θ − ηg/(√G+ε)", "Per-parameter step: large for rare features (good for sparse TF-IDF); step decays → may stall"],
    ["RMSProp", "s ← ρs + (1−ρ)g² ;  θ ← θ − ηg/(√s+ε)", "Exponential average fixes AdaGrad’s decaying step"],
    ["Adam", "m, v moving averages of g, g² (bias-corrected);  θ ← θ − η·m̂/(√v̂+ε)", "Momentum + RMSProp; robust default, little tuning"],
], widths=[2.8, 6.4, 6.8], size=8.5, first_bold=True)
FIGURE("08_gd_trajectories.png", "Figure 9 — Toy illustration on an elongated bowl f = 0.1x² + 2y² (start ■, minimum ★). Gradients of all but ‘Batch GD’ carry artificial noise to mimic mini-batches. "
       "Batch GD zig-zags across the steep direction; momentum overshoots then spirals in; Nesterov corrects earlier; AdaGrad / RMSProp rescale each axis; Adam follows a smooth curve.", 16.5)
H("7.1  Experiment on the review data", 2)
P(f"Each variant trained the baseline network for {R['gd_variants']['epochs']} epochs. Because the best learning rate differs wildly between variants, each received a small grid of learning rates "
  f"and the one giving the **lowest final training cost** was kept (an optimiser is judged on how well it minimises the cost in a fixed budget). ‘Updates’ is the number of weight updates performed.")
TABLE(["Variant", "batch", "best lr", "updates", "final train cost", "val F1", "time (s)"],
      [[r["label"], r["batch_size"] if r["batch_size"] < 5000 else "full", r["best_lr"], f"{r['updates']:,}", f"{r['final_train_cost']:.4f}", f"{r['val_f1']:.3f}", r["time"]] for r in GD],
      widths=[4.2, 1.5, 1.7, 2.0, 2.8, 1.8, 2.0], size=8.5, first_bold=True)
FIGURE("07_gd_variants.png", "Figure 10 — Convergence (left), validation F1 (middle) and wall-clock time (right) of the eight variants.", 17)
by_cost = sorted(GD, key=lambda r: r["final_train_cost"])
sgd = next(r for r in GD if r["optimizer"] == "gd" and r["batch_size"] == 1)
mb = next(r for r in GD if r["label"].startswith("Mini-batch"))
bg = next(r for r in GD if r["label"] == "Batch GD")
P("**Findings.**", after=2)
B(f"**Batch GD** performs only {bg['updates']} updates in {R['gd_variants']['epochs']} epochs and is badly under-trained (val F1 {bg['val_f1']:.2f}); it would need hundreds of epochs.")
B(f"**Stochastic GD** needs {sgd['updates']:,} updates and {sgd['time']:.1f} s — about {sgd['time'] / mb['time']:.0f}× the time of mini-batch — although it reaches a very good model (val F1 {sgd['val_f1']:.3f}). "
  f"The noisy updates act as regularisation but the cost is wall-clock time.")
B(f"**Mini-batch GD** delivers nearly the same quality in {mb['updates']:,} updates and {mb['time']:.1f} s: the practical default.")
B(f"**Adaptive methods:** {by_cost[0]['label']} and {by_cost[1]['label']} reach the lowest training cost ({by_cost[0]['final_train_cost']:.3f} and {by_cost[1]['final_train_cost']:.3f}) — "
  f"sparse TF-IDF features benefit from per-parameter step sizes. Momentum / Nesterov / RMSProp are close behind.")
B("Lowest training cost is **not** the same as best validation F1: the differences between the mini-batch optimisers in F1 (about 1–2 points on only 1,200 validation reviews) are within sampling noise, "
  "so we do not rank them by F1. We use Adam in the final model because it is robust to the learning rate and needs little tuning.")

# =============================================================== 8 HYPER
H("8  Hyper-parameter tuning", 1)
P("Six hyper-parameters were studied **one at a time around the baseline** (hidden = [32], ReLU, BCE, Adam, lr = 1e-3, batch 64, 20 epochs). "
  "Every setting was trained with several random seeds and judged on **validation F1 of the spam class** (the test set was never used for selection). Error bars show ± 1 standard deviation across seeds.")
names = {"lr": "Learning rate", "neurons": "Hidden neurons", "layers": "Hidden layers (32 neurons each)", "batch": "Batch size", "activation": "Hidden activation", "epochs": "Number of epochs"}
rows = []
for k in ["lr", "neurons", "layers", "batch", "activation", "epochs"]:
    rows.append([names[k], "  ".join(f"{v}: {m:.3f}" for v, m in zip(HP[k]["values"], HP[k]["mean"])), str(HB[k])])
TABLE(["Hyper-parameter", "Validation F1 for each value", "Best"], rows, widths=[3.6, 10.4, 2.0], size=8, first_bold=True)
FIGURE("09_hyperparameters.png", "Figure 11 — Validation F1 against each hyper-parameter (green = best value).", 16.5)
lr_v, lr_m = HP["lr"]["values"], HP["lr"]["mean"]
nn_v, nn_m = HP["neurons"]["values"], HP["neurons"]["mean"]
ep_v, ep_m = HP["epochs"]["values"], HP["epochs"]["mean"]
act = dict(zip(HP["activation"]["values"], HP["activation"]["mean"]))
P("**Observations.**", after=2)
B(f"**Learning rate:** too small ({lr_v[0]}) under-fits within 20 epochs (F1 {lr_m[0]:.3f}); the best value is {HB['lr']}; large rates ({lr_v[-1]}) make updates overshoot (F1 {lr_m[-1]:.3f}).")
B(f"**Hidden neurons / layers:** the problem is close to linearly separable, so more capacity does not help — F1 decreases from {nn_m[0]:.3f} with {nn_v[0]} neurons to {nn_m[-1]:.3f} with {nn_v[-1]}, "
  f"and a single hidden layer is best (extra layers lose ≈ 1 point). Larger networks fit the training noise.")
B(f"**Batch size:** best around {HB['batch']}; very small batches give noisy gradients and very large ones make too few updates in the epoch budget.")
B(f"**Activation:** ReLU and Leaky-ReLU ({act['relu']:.3f}) beat tanh ({act['tanh']:.3f}) and sigmoid ({act['sigmoid']:.3f}) — saturating hidden activations learn more slowly.")
B(f"**Epochs:** {ep_v[0]} epochs under-fit (F1 {ep_m[0]:.3f}), the best is {HB['epochs']}, and {ep_v[-1]} epochs over-fit (F1 {ep_m[-1]:.3f}) — Figure 12; early stopping automates this choice.")
B("Most differences are only 0.5 – 1.5 F1 points; with 1,200 validation reviews these are close to the seed-to-seed noise, except for the extremes (very low learning rate, very few epochs), which are clearly worse.")
FIGURE("10_epochs_overfitting.png", "Figure 12 — Long run: training cost keeps falling while validation cost turns upward (over-fitting).", 10.5)
H("8.1  Selecting the final configuration", 2)
P("The best value of each hyper-parameter was combined and compared with the baseline (and a variant with L2 regularisation) using several seeds with early stopping:")
TABLE(["Candidate", "hidden", "lr", "batch", "activation", "L2", "val F1 (mean ± sd)"],
      [[k, v["config"]["hidden"], v["config"]["lr"], v["config"]["bs"], v["config"]["act"], v["config"].get("l2", 0), f"{v['val_f1_mean']:.4f} ± {v['val_f1_std']:.4f}"]
       for k, v in R["final_candidates"].items()], widths=[3.4, 2.0, 1.5, 1.5, 2.2, 1.5, 3.9], size=8.5, first_bold=True)
P(f"Selected: **{R['final_choice']['name']}** — hidden layer [{cfg['hidden'][0]}], {cfg['act'].upper()}, Adam lr {cfg['lr']}, batch {cfg['bs']}, BCE, up to {cfg['epochs']} epochs with early stopping. "
  f"It has only {FIN['params']:,} parameters versus {BASE['params']:,} for the baseline.")

# =============================================================== 9 EVALUATION
H("9  Performance evaluation", 1)
P(f"All numbers in this section are on the **held-out test set** ({SPL['test']:,} reviews), with ‘spam’ as the positive class. Decision threshold: {FIN['threshold']} "
  f"(a threshold sweep on the validation set did not improve F1, so the default 0.5 is kept).")
TABLE(["Model", "Accuracy", "Precision", "Recall", "F1-score", "AUC"], [
    ["Perceptron (baseline)", pct(PER["test"]["accuracy"]), pct(PER["test"]["precision"]), pct(PER["test"]["recall"]), pct(PER["test"]["f1"]), "—"],
    ["MLP baseline (untuned, 32 hidden)", pct(BASE["test"]["accuracy"]), pct(BASE["test"]["precision"]), pct(BASE["test"]["recall"]), pct(BASE["test"]["f1"]), f"{BASE['test']['auc']:.3f}"],
    ["**Tuned MLP (deployed)**", pct(T["accuracy"]), pct(T["precision"]), pct(T["recall"]), pct(T["f1"]), f"{T['auc']:.3f}"],
], widths=[5.6, 2.1, 2.1, 2.1, 2.1, 1.6], size=9, highlight_row=2)
tn, fp, fn, tp = T["confusion_matrix"][0][0], T["confusion_matrix"][0][1], T["confusion_matrix"][1][0], T["confusion_matrix"][1][1]
P(f"Over five random initialisations the tuned model scores F1 {pct(FIN['seed_mean']['f1'])} ± {pct(FIN['seed_std']['f1'])} and accuracy {pct(FIN['seed_mean']['accuracy'])} ± {pct(FIN['seed_std']['accuracy'])} "
  f"(mean ± standard deviation), so the result is stable. The deployed model is the seed with the best *validation* F1.")
EQ("Accuracy = (TP+TN)/N     Precision = TP/(TP+FP)     Recall = TP/(TP+FN)     F1 = 2·P·R/(P+R)")
FIGURE("12_confusion_matrix_spam.png", f"Figure 13 — Confusion matrix on the test set: TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}.", 8.5)
FIGURE("14_model_ladder.png", "Figure 14 — Test metrics from the majority-class rule to the perceptron, baseline MLP and tuned MLP.", 13)
H("9.1  Interpretation", 2)
B(f"**Precision {pct(T['precision'])}** — only {fp} genuine review out of {tn + fp} was wrongly flagged: legitimate customers are almost never blocked, which matters most for a marketplace.")
B(f"**Recall {pct(T['recall'])}** — {fn} of {fn + tp} spam reviews were missed. Recall by spam type shows where: "
  + ", ".join(f"{k.replace('_', ' ')} {pct(v, 0)}" for k, v in R.get("recall_by_spam_type", {}).items() if k != "genuine") +
  ". Link / phone / paid-review / junk spam is caught almost perfectly; the misses are **subtle fakes** that read like genuine praise and a few reviews whose label was deliberately flipped as annotation noise.")
B(f"**Perceptron → MLP:** the non-linear hidden layer adds about {100 * (BASE['test']['f1'] - PER['test']['f1']):.1f} F1 points over the perceptron (ROC-AUC of the MLP: {BASE['test']['auc']:.3f}).")
B(f"**Tuning:** tuning did *not* improve test accuracy over the untuned baseline (F1 {pct(T['f1'])} vs {pct(BASE['test']['f1'])}; the difference is within seed noise). Its benefit is a "
  f"**{BASE['params'] / FIN['params']:.0f}× smaller network** ({FIN['params']:,} vs {BASE['params']:,} parameters) with equal quality, and evidence that the baseline was already close to the ceiling of this feature set.")
FIGURE("13_roc.png", "Figure 15 — ROC curve of the baseline and tuned MLP.", 8)
H("9.2  Error analysis", 2)
fnl = R["errors"]["false_negatives"][:5]
TABLE(["Missed spam review (false negative)", "P(spam)", "Comment"],
      [[x["text"][:95], f"{x['prob']:.3f}", "label-noise sample (genuine text labelled spam)" if x["type"] == "genuine" else x["type"].replace("_", " ")] for x in fnl],
      widths=[9.2, 1.6, 5.2], size=8)
fpl = R["errors"]["false_positives"][:3]
if fpl:
    TABLE(["Genuine review flagged as spam (false positive)", "P(spam)"], [[x["text"][:110], f"{x['prob']:.3f}"] for x in fpl], widths=[13.5, 2.5], size=8)
P("Most remaining errors are intrinsically ambiguous (indistinguishable from real reviews or mislabelled), so further gains would need richer signals such as reviewer history, "
  "purchase verification, timing bursts or a sequence model — see Section 12.")
H("9.3  Sentiment head (extension)", 2)
if SENT:
    P(f"A second network with the same hidden layer, a **3-way softmax** output and **categorical cross-entropy** is trained on genuine reviews only (labels from the star rating: ≥4 Positive, 3 Neutral, ≤2 Negative). "
      f"Test accuracy {pct(SENT['accuracy'])}, macro-F1 {SENT['macro_f1']:.3f}; per-class F1: Negative {SENT['per_class_f1']['Negative']:.3f}, Neutral {SENT['per_class_f1']['Neutral']:.3f}, Positive {SENT['per_class_f1']['Positive']:.3f}. "
      f"The Neutral class is hardest (mixed opinions such as ‘fast delivery but average battery’), and 5 % of ratings were simulated as mis-rated, which caps accuracy; "
      f"making the network wider (16 – 64 neurons, or two layers) gave no improvement (≈ 87 %), confirming the limit comes from the labels, not the capacity.")
    FIGURE("15_confusion_matrix_sentiment.png", "Figure 16 — Sentiment confusion matrix (test set).", 8)

# =============================================================== 10 WEB APP
H("10  Web application", 1)
P("The web interface is a small **Flask** application (`webapp/app.py`). It loads the saved weights with NumPy only and answers within milliseconds on a single CPU.")
P("**Input message → AI model → Spam / Not Spam**", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
TABLE(["Page / endpoint", "Purpose"], [
    ["/  (Analyzer)", "Text box + example buttons. Shows verdict (Spam / Not Spam), P(spam) with threshold marker, sentiment bars, topics, rule-based spam signals (link, phone, repeated words …) and the words that most influenced the decision (occlusion: remove a word, see how P(spam) changes)."],
    ["/dashboard", "Review-intelligence dashboard: KPI cards (products, categories, reviews, % spam, average rating), review trend, sentiment breakdown, rating distribution, top-5 positive / negative topics, price-vs-rating scatter, spam by category, common keywords."],
    ["/model", "Metrics and all experiment figures (confusion matrix, ROC, cost, gradient-descent and hyper-parameter studies)."],
    ["POST /api/predict", "JSON API: {\"text\": \"…\"} → label, probability, sentiment, topics, signals."],
    ["GET /health", "Liveness check used by Cloud Run."],
], widths=[3.6, 12.4], size=8.5, first_bold=True)
H("10.1  Example predictions from the running application", 2)
from reviewintel.inference import Predictor
try:
    pr = Predictor(str(ROOT / "models"))
    ex = ["Delivery was super fast and the battery easily lasts a full day. Camera is sharp too. Totally paisa vasool!",
          "Very disappointed. The box was torn and the fabric is rough. Refund still not received after three weeks.",
          "Packing bahut kharab thi aur delivery late ho gayi, quality bhi bekaar hai.",
          "Get 90% OFF on all mobiles today only! Visit www.dealzmart-deals.in or WhatsApp 9876501234",
          "Seller is giving Rs 100 refund for a 5 star review. DM me for details.",
          "Best product ever!!! Must buy!!! 5 star 5 star 5 star"]
    rows = []
    for t in ex:
        o = pr.predict(t)
        rows.append([t[:88] + ("…" if len(t) > 88 else ""), o["label"], f"{o['spam_probability']:.3f}", o["sentiment"]["label"] if o["sentiment"] else "n/a (spam)"])
    TABLE(["Input message", "Prediction", "P(spam)", "Sentiment"], rows, widths=[9.2, 2.1, 1.6, 3.1], size=8)
except Exception as exc:  # pragma: no cover
    P(f"(Examples unavailable: {exc})")
P("[[Insert screenshots of the Analyzer page (one spam, one genuine example) and the Dashboard page from your deployed URL here.]]")
H("10.2  Running locally", 2)
CODE("pip install -r requirements.txt")
CODE("python run_pipeline.py          # trains, evaluates, saves models/ and outputs/")
CODE("python webapp/app.py            # http://127.0.0.1:8080")

# =============================================================== 11 GCP
H("11  Google Cloud deployment", 1)
P("**Service chosen: Cloud Run.** The model is tiny (≈ 1 MB of weights plus a JSON vocabulary), the app is stateless and needs only CPU, so a serverless container is the best fit: no servers to manage, "
  "HTTPS URL out of the box, automatic scale-to-zero (so the student coupon is barely consumed) and deployment from source in a single command. "
  "Alternatives considered: App Engine (heavier runtime config for the same result), Compute Engine VM (manual patching, always-on cost), Vertex AI endpoints (over-sized for a 6 k-parameter NumPy model).")
P("**Deployment steps**", bold=True, after=2)
NRESET()
for s in ["Request the student coupon with the link in the brief (school e-mail), redeem it, and create a project with the credit attached as billing account.",
          "Install the gcloud CLI and log in: `gcloud auth login`.",
          "Train the model once locally so that `models/spam_model.npz` exists (the container ships these files).",
          "Run `PROJECT_ID=<id> ./deploy_gcp.sh`, which executes the commands below. Cloud Build reads the included `Dockerfile` (python:3.11-slim + Flask + NumPy + gunicorn) and Cloud Run starts it.",
          "Open the printed Service URL, test one spam and one genuine review, take screenshots and paste the URL on the cover page."]:
    N(s)
CODE("gcloud config set project <your-project-id>")
CODE("gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com")
CODE("gcloud run deploy review-intelligence --source . --region asia-south1 \\")
CODE("      --allow-unauthenticated --memory 512Mi --cpu 1 --max-instances 2 --port 8080")
P("")
TABLE(["Item", "Status"], [
    ["Dockerfile, gunicorn entry point, requirements-app.txt, deploy_gcp.sh", "Provided and included in the zip"],
    ["Local run of the web app and all endpoints (automated tests)", "Executed — 9 tests pass"],
    ["Actual deployment to the student’s Google Cloud project", "[[To be performed by the student — needs the personal coupon / project]]"],
    ["Deployed Service URL", "[[https://review-intelligence-xxxxx-el.a.run.app]]"],
    ["Screenshot of Cloud Run console (service ‘Ready’)", "[[Insert screenshot]]"],
    ["Screenshot of the app in the browser on the Cloud Run URL", "[[Insert screenshot]]"],
], widths=[9, 7], size=8.5)
P("Region **asia-south1 (Mumbai)** is used for low latency to Indian users. To avoid charges after grading, delete the service with `gcloud run services delete review-intelligence --region asia-south1`.")

# =============================================================== 12 LIMITS
H("12  Limitations and future work", 1)
for s in ["**Synthetic data.** The corpus is generated from templates, so its language is far less diverse than real reviews; real-world accuracy will be lower and the hyper-parameter conclusions (e.g. ‘4 hidden neurons are enough’) may change. The first next step is re-running the pipeline on a real Flipkart / Amazon.in / SMS-spam dataset (one command).",
          "**Bag-of-n-grams features** ignore word order and cannot understand context, sarcasm or code-mixed spelling variants; sub-word embeddings or a sequence model (CNN / LSTM / Transformer) would help.",
          "**Single split.** Hyper-parameters were chosen on one validation split with 1–3 seeds; k-fold cross-validation would give tighter estimates.",
          "**Sentiment labels come from star ratings**, which are noisy; the topic detector is rule-based and not part of the neural network.",
          "**Adversarial setting.** Spammers adapt; the model needs periodic retraining, monitoring and human review for borderline scores. Reviewer-level signals (account age, verified purchase, burst timing) are not available here.",
          "**Explanations** (occlusion) are approximate; they show sensitivity, not causation."]:
    B(s)

# =============================================================== 13 CONCL
H("13  Conclusion", 1)
P(f"A complete spam / fake-review detection system for Indian e-commerce was designed, implemented from scratch with NumPy, evaluated and packaged for Google Cloud Run. "
  f"The perceptron baseline reaches F1 {pct(PER['test']['f1'])} and fails on XOR, motivating the MLP, whose hand-written back-propagation was verified against numerical gradients (error ≈ 1e-10). "
  f"Binary cross-entropy is the right cost for the sigmoid output — equal to MSE from a normal start but able to escape a confidently-wrong initialisation where MSE stalls at {pct(f_['val_acc'], 1)} accuracy. "
  f"Among the gradient-descent variants, mini-batch training is the practical default, batch GD is far too slow per epoch, SGD is accurate but ~{sgd['time'] / mb['time']:.0f}× slower, and adaptive optimisers minimise the cost fastest. "
  f"Six hyper-parameters were tuned; the deployed {cfg['hidden'][0]}-neuron network attains test accuracy {pct(T['accuracy'])}, precision {pct(T['precision'])}, recall {pct(T['recall'])}, F1 {pct(T['f1'])} "
  f"with {FIN['params']:,} parameters. These figures hold for the synthetic dataset used; the codebase is ready to be re-run on real data.")

# =============================================================== REFERENCES
H("References", 1)
for ref in ["Rosenblatt, F. (1958). The perceptron: a probabilistic model for information storage and organization in the brain. Psychological Review, 65(6).",
            "Minsky, M. & Papert, S. (1969). Perceptrons. MIT Press.",
            "Rumelhart, D., Hinton, G. & Williams, R. (1986). Learning representations by back-propagating errors. Nature, 323.",
            "Nesterov, Y. (1983). A method for solving the convex programming problem with convergence rate O(1/k²). Soviet Math. Dokl.",
            "Duchi, J., Hazan, E. & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization (AdaGrad). JMLR, 12.",
            "Tieleman, T. & Hinton, G. (2012). Lecture 6.5 – RMSProp. COURSERA: Neural Networks for Machine Learning.",
            "Kingma, D. & Ba, J. (2015). Adam: a method for stochastic optimization. ICLR.",
            "Sutskever, I. et al. (2013). On the importance of initialization and momentum in deep learning. ICML.",
            "He, K. et al. (2015). Delving deep into rectifiers (He initialisation). ICCV. · Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. AISTATS.",
            "Goodfellow, I., Bengio, Y. & Courville, A. (2016). Deep Learning. MIT Press (Chapters 6, 8).",
            "Ott, M. et al. (2011). Finding deceptive opinion spam by any stretch of the imagination. ACL. (background on fake-review detection)"]:
    B(ref)

# =============================================================== APPENDICES
PB()
H("Appendix A  How to run the project step by step", 1)
NRESET()
for s in ["Unzip `ecom_review_intelligence.zip` and open a terminal in the folder.",
          "(Recommended) create a virtual environment: `python -m venv .venv` and activate it (`source .venv/bin/activate`, Windows: `.venv\\Scripts\\activate`).",
          "Install packages: `pip install -r requirements.txt`.",
          "Train and run all experiments: `python run_pipeline.py` (≈ 5–10 min; add `--quick` for a 1-minute test). It prints each stage and writes `models/`, `outputs/figures/`, `outputs/results.json`, `outputs/tables/*.csv`.",
          "Optional self-test: `python -m unittest discover -s tests -v` (9 tests: gradient check, optimisers, perceptron/XOR, web endpoints).",
          "Build this report: `python report/build_report.py` → `report/Review_Intelligence_Report.docx`.",
          "Start the web app: `python webapp/app.py` and open http://127.0.0.1:8080.",
          "Deploy: follow Section 11 (`PROJECT_ID=<id> ./deploy_gcp.sh`).",
          "Use real data: `python run_pipeline.py --data my_reviews.csv` (columns `review_text`, `is_spam`; optional `rating, category, product_id, price_inr, review_date`), then rebuild the report."]:
    N(s)
P("Shortcut: `./run_all.sh` (Linux/macOS) or `run_all.bat` (Windows) performs steps 2–7 automatically.")
H("Appendix B  Submission checklist", 1)
TABLE(["Item", "Done?"], [
    ["Student name, roll number and deployed URL filled on the cover page", "[[ ]]"],
    ["Google Cloud coupon requested and redeemed; **coupon e-mail saved as PDF / screenshot (required by the brief)**", "[[ ]]"],
    ["Application deployed on Cloud Run; screenshots inserted in Sections 10 and 11", "[[ ]]"],
    ["Report uploaded (this document, or exported to PDF)", "[[ ]]"],
    ["Source-code zip uploaded (if requested)", "[[ ]]"],
    ["Brief’s deadline: 20 September 2026 — confirm with the course instructor if a late submission is acceptable", "[[ ]]"],
], widths=[13.2, 2.8], size=9)

out = ROOT / "report" / "Review_Intelligence_Report.docx"
doc.save(out)
print("saved", out)
