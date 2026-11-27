import json, pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont("DV", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVI", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")
ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["BodyText"], fontName="DV", fontSize=10.5, leading=14.5, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=B, fontName="DVB", fontSize=17, leading=21, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=B, fontName="DVB", fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=4)
SM = ParagraphStyle("sm", parent=B, fontSize=9, leading=12, textColor=colors.HexColor("#555555"))
R = "results/"
main = pd.read_csv(R + "main_results.csv"); cc = pd.read_csv(R + "cross_category.csv")
abl = pd.read_csv(R + "stylometric_ablation.csv"); summ = json.load(open(R + "summary.json"))
top = json.load(open(R + "top_ngrams.json")); bylen = pd.read_csv(R + "f1_by_length.csv")
cm_ = summ["confusion_matrix_controlled_word_lr"]
RR = "results_ru/"; ru = pd.read_csv(RR + "main_results.csv"); ru_s = json.load(open(RR + "summary.json"))
ru_loo = pd.read_csv(RR + "leave_one_generator_out.csv"); ru_ho = pd.read_csv(RR + "heldout_claude.csv")
ru_pg = pd.read_csv(RR + "per_generator_recall.csv").set_index("generator").recall
rg = lambda st, m, c="macro_f1_mean": ru[(ru.setting == st) & (ru.model == m)][c].iloc[0]
hg = lambda st, m, c: ru_ho[(ru_ho.setting == st) & (ru_ho.model == m)][c].iloc[0]
DE, DR = json.load(open("results/defense_report_en.json")), json.load(open("results_ru/defense_report_ru.json"))
de, dr = pd.read_csv("results/defense_results_en.csv"), pd.read_csv("results_ru/defense_results_ru.csv")

def tbl(rows, widths, header=True):
    t = Table(rows, colWidths=widths)
    st = [("FONT", (0, 0), (-1, -1), "DV", 9), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header: st += [("FONT", (0, 0), (-1, 0), "DVB", 9), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF1FB"))]
    t.setStyle(TableStyle(st)); return t

g = lambda s, m, c="macro_f1": main[(main.setting == s) & (main.model == m)][c].iloc[0]
names = {"stylometric": "Stylometric features + LR", "tfidf_word_lr": "Word TF-IDF (1–2-grams) + LR",
         "tfidf_char_lr": "Char TF-IDF (2–5-grams) + LR", "tfidf_word_svm": "Word TF-IDF + Linear SVM",
         "combined": "Combined + SVM"}
P = []
P += [Paragraph("Detecting Computer-Generated Product Reviews in English and Russian with Classical NLP", H1),
      Paragraph("NLP2 mini-project · e-commerce review integrity · November 2026", SM), Spacer(1, 6)]

P += [Paragraph("Abstract", H2), Paragraph(
    f"<b>Final result (Section 6):</b> with hyperparameters chosen by cross-validation on the training split only and a single "
    f"evaluation on a held-out test split, a word/lemma + character TF-IDF linear SVM reaches macro-F1 <b>{de.f1_macro.iloc[-1]:.3f}</b> "
    f"in English ({int(de.test_n.iloc[-1]):,} test reviews; stylometric baseline {de.f1_macro.iloc[1]:.3f}, majority class {de.f1_macro.iloc[0]:.3f}) "
    f"and <b>{dr.f1_macro.iloc[-1]:.3f}</b> in Russian ({int(dr.test_n.iloc[-1])} test reviews; baseline {dr.f1_macro.iloc[1]:.3f}). "
    f"Sections 3–5 report the exploratory analyses that led to it. "
    f"We train classical (non-LLM) classifiers to separate GPT-2-generated product reviews from real Amazon reviews "
    f"on the Salminen et al. (2022) dataset ({summ['n_reviews']:,} reviews, 10 categories, balanced). "
    f"A combined word + character TF-IDF and stylometric model reaches <b>macro-F1 {g('raw','combined'):.3f}</b> "
    f"(ROC-AUC {g('raw','combined','roc_auc'):.3f}). We find a strong dataset shortcut: only "
    f"{summ['end_punct_rate']['CG']*100:.0f}% of generated reviews end with punctuation versus "
    f"{summ['end_punct_rate']['OR']*100:.0f}% of real ones, because generation was cut off mid-sentence. Removing it costs the "
    f"stylometric model {100*(g('raw','stylometric')-g('artifact-controlled','stylometric')):.1f} F1 points but barely affects n-gram models. "
    f"Under a size-matched test on unseen product categories, F1 drops by about {abs(cc.gap.mean())*100:.0f} points on average, "
    f"most of all for Movies & TV. "
    f"<b>Russian extension:</b> we built a Russian dataset (real RuReviews plus reviews generated with ruGPT-3 and Qwen2.5). "
    f"The same pipeline with lemmatization reaches macro-F1 {rg('controlled','combined'):.2f} on those generators, "
    f"but the final model catches only {DR['modern_llm_probe']['fake_recall']*100:.0f}% of held-out reviews written by a modern LLM (Claude). "
    f"The English model is at chance on Russian (AUC {ru_s['en_model_on_ru']['roc_auc']:.2f}).", B)]

P += [Paragraph("1. Research questions", H2), Paragraph(
    "<b>RQ1.</b> How well do classical NLP features detect machine-generated reviews, and which features carry the signal?<br/>"
    "<b>RQ2.</b> How much of that performance comes from generation artifacts rather than writing style?<br/>"
    "<b>RQ3.</b> Does a detector trained on some product categories work on a category it has never seen?", B)]

P += [Paragraph("2. Data and setup", H2), Paragraph(
    f"<b>Dataset:</b> Salminen et al. (2022), “Creating and detecting fake reviews of online products”, J. Retailing and Consumer Services. "
    f"Label CG = generated by a fine-tuned GPT-2, OR = original Amazon review. After removing empty and duplicate texts: "
    f"{summ['n_reviews']:,} reviews. Generated reviews are shorter on average ({summ['mean_words']['CG']:.0f} vs "
    f"{summ['mean_words']['OR']:.0f} words); ratings are balanced (mean 4.26 vs 4.25).<br/>"
    f"<b>Split:</b> stratified 80/20 ({summ['train']:,} train / {summ['test']:,} test), seed 42.<br/>"
    "<b>Artifact-controlled setting:</b> whitespace collapsed and trailing punctuation removed from every review; "
    "the stylometric features ends_with_punct and double_space are dropped.<br/>"
    "<b>Metrics:</b> macro-F1 (main), accuracy, ROC-AUC; confusion matrix; F1 by review length.", B)]

P += [Paragraph("3. Results", H2), Paragraph("<b>3.1 Model comparison (RQ1, RQ2)</b>", B), Paragraph("“Combined” = word + char TF-IDF + stylometric features. “ctrl.” = artifact-controlled.", SM)]
rows = [["Model", "F1 raw", "AUC raw", "F1 ctrl.", "AUC ctrl."]]
for m in names:
    rows.append([names[m], f"{g('raw',m):.3f}", f"{g('raw',m,'roc_auc'):.3f}",
                 f"{g('artifact-controlled',m):.3f}", f"{g('artifact-controlled',m,'roc_auc'):.3f}"])
P += [tbl(rows, [6.6*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm]), Spacer(1, 6),
      Image(R + "fig_models.png", width=15*cm, height=7.7*cm),
      Paragraph(
    "N-gram models are by far the strongest. Word TF-IDF scores are identical in both settings because its tokenizer "
    "ignores punctuation and whitespace, so it never saw the shortcut. Character n-grams and the combined model see "
    "punctuation and lose a little (0.963 → 0.960). The hand-built stylometric model relies on the artifact most heavily.", B)]

P += [Paragraph("<b>3.2 What signals fakes?</b>", B), Paragraph(
    f"<b>Stylometric ablation</b> (drop one feature, raw setting): removing ends_with_punct costs "
    f"{abs(abl.iloc[0].delta)*100:.1f} F1 points, while every other feature costs under 2 points. That makes it the shortcut. "
    "The genuine style signals that remain are type-token ratio (generated text repeats words) and function-word ratio.<br/>"
    f"<b>Top n-grams (controlled word LR).</b> Most indicative of fakes: {', '.join(top['fake'][:12])}. "
    f"Most indicative of real reviews: {', '.join(top['real'][:12])}.<br/>"
    "Real reviews lean on contrast and cause markers (<i>though, however, because, even, without, after</i>), which "
    "suggests humans build more argument structure: what happened, why, and the catch. Generated reviews use "
    "generic, repetitive phrasing.", B)]
lr = [["Review length (words)"] + bylen.len_bucket.tolist(), ["Macro-F1"] + [f"{v:.3f}" for v in bylen.macro_f1]]
P += [tbl(lr, [4.4*cm] + [2.4*cm]*len(bylen)), Spacer(1, 4), Paragraph(
    f"Short reviews are the hardest (F1 {bylen.macro_f1.iloc[0]:.3f} at 20 words or fewer). "
    f"Confusion matrix (controlled word LR, rows = true real / fake): [[{cm_[0][0]}, {cm_[0][1]}], [{cm_[1][0]}, {cm_[1][1]}]]. "
    "Fakes slip through more often than real reviews get flagged.", B)]

P += [Paragraph("<b>3.3 Generalization to unseen categories (RQ3)</b>", B), Paragraph(
    "For each category we hold out the same 20% test set and train word TF-IDF + LR three ways: (a) on that category, "
    "(b) on the <i>same number</i> of reviews from the other nine categories, and (c) on all other-category data (~36k). "
    "Comparison (b) is the fair one, because (c) also gets 10 times more data.", B),
      Image(R + "fig_cross_category.png", width=15*cm, height=9*cm)]
rows = [["Held-out category", "Same cat.", "Other, matched", "Other, all", "Gap (b–a)"]]
for _, r in cc.sort_values("gap").iterrows():
    rows.append([r.held_out.replace("_", " "), f"{r.in_domain_f1:.3f}", f"{r.cross_f1_size_matched:.3f}", f"{r.cross_f1_all_data:.3f}", f"{r.gap:+.3f}"])
rows.append(["Mean", f"{cc.in_domain_f1.mean():.3f}", f"{cc.cross_f1_size_matched.mean():.3f}", f"{cc.cross_f1_all_data.mean():.3f}", f"{cc.gap.mean():+.3f}"])
P += [tbl(rows, [5.4*cm, 2.2*cm, 3.4*cm, 2.4*cm, 2.2*cm]), Spacer(1, 4), Paragraph(
    f"With equal training sizes, category shift costs {abs(cc.gap.mean())*100:.1f} F1 points on average. Movies & TV "
    "suffers most, since its reviews talk about plot and characters rather than product features. More out-of-domain data "
    "closes most, but not all, of the gap.", B)]

P += [Paragraph("4. Error analysis", H2), Paragraph(
    "From a 40-error sample (results/errors_sample.csv): (1) <b>Fluent short fakes</b> pass as real, e.g. “These work REALLY "
    "well. I use it as a night light…”; they are too short to show repetition. (2) <b>Narrative fakes</b> in Movies/Books "
    "mimic human emotion (“It was agony for me… But. I liked him.”). (3) <b>Real reviews with typos or odd phrasing</b> "
    "get flagged as fake (“the qaulity for the price”, “So inconclusion i liked this book”), so the model partly confuses "
    "non-standard English with machine text. That is a fairness risk for non-native reviewers.", B)]


P += [Paragraph("5. Russian-language extension", H2), Paragraph(
    "<b>Why retrain?</b> The English model does not transfer. Applied to the Russian data it scores macro-F1 "
    f"{ru_s['en_model_on_ru']['macro_f1']:.2f} and ROC-AUC {ru_s['en_model_on_ru']['roc_auc']:.2f} (chance level), because "
    f"{ru_s['en_model_on_ru']['share_all_zero_rows']*100:.0f}% of Russian reviews contain no word the English model knows.", B),
  Paragraph(
    "<b>Data.</b> Real: RuReviews (Smetanin &amp; Komarov, 2019), reviews of women's clothing and accessories from a Russian online store. "
    "Fake: we generated reviews conditioned on the sentiment and length of randomly drawn real reviews, using two open models: "
    "<b>ruGPT-3 small</b> (ai-forever, a GPT-2-era model, comparable to the English dataset) and <b>Qwen2.5-0.5B-Instruct</b> "
    "(a modern instruction-tuned model). We removed non-Cyrillic outputs (Qwen sometimes switched to Chinese) and refusal or meta "
    f"text, keeping {ru_s['n_kept_after_filter']['rugpt3']} ruGPT-3 and {ru_s['n_kept_after_filter']['qwen']} Qwen reviews, "
    f"matched by sentiment with {DR['split']['train'] + DR['split']['test'] - ru_s['n_kept_after_filter']['rugpt3'] - ru_s['n_kept_after_filter']['qwen']:,} real reviews. A third set of {ru_s['stats']['claude']['n']} reviews written "
    "by Claude (a frontier LLM) is used <b>only as a held-out test</b> and is never trained on.", B),
  Paragraph(
    "<b>Method changes.</b> Lemmatization with pymorphy3 (Russian inflection splits one word into many forms), a Russian function-word list, "
    "and a stricter <i>controlled</i> setting (lowercase, all punctuation removed). Real reviews are often lowercase and unpunctuated "
    f"(only {ru_s['stats']['real']['starts_upper']*100:.0f}% start with a capital letter, versus {ru_s['stats']['qwen']['starts_upper']*100:.0f}% for Qwen), "
    "a shortcut similar to the English truncation artifact. All scores use 5-fold cross-validation (mean ± std).", B)]
rows = [["Model", "F1 raw", "F1 controlled", "AUC controlled"]]
for m, nm in [("stylometric", "Stylometric + LR"), ("word", "Word TF-IDF + LR"), ("lemma", "Lemma TF-IDF + LR"), ("char", "Char TF-IDF + LR"), ("combined", "Combined + SVM")]:
    rows.append([nm, f"{rg('raw',m):.3f} ± {rg('raw',m,'macro_f1_std'):.3f}", f"{rg('controlled',m):.3f} ± {rg('controlled',m,'macro_f1_std'):.3f}", f"{rg('controlled',m,'roc_auc_mean'):.3f}"])
P += [tbl(rows, [5.2*cm, 3.6*cm, 3.6*cm, 3*cm]), Spacer(1, 4), Image(RR + "fig_models_ru.png", width=14*cm, height=7.2*cm),
  Paragraph(
    f"Lemmatization helps slightly over raw words ({rg('controlled','word'):.3f} → {rg('controlled','lemma'):.3f}). Stylometry depends heavily on "
    f"casing and punctuation: it drops from {rg('raw','stylometric'):.2f} to {rg('controlled','stylometric'):.2f} once they are removed. Recall by generator "
    f"(combined, controlled): ruGPT-3 {ru_pg['rugpt3']:.2f}, Qwen {ru_pg['qwen']:.2f}; real reviews correctly kept {ru_pg['real (specificity)']:.2f}.", B),
  Paragraph(
    f"<b>What the model learned.</b> Lemmas most indicative of fakes: {', '.join(ru_s['top_lemmas']['fake'][:12])}. "
    f"Most indicative of real reviews: {', '.join(ru_s['top_lemmas']['real'][:12])}. Real buyers write about concrete experience "
    "(size, seams, synthetic fabric, height, the seller, returning the item). Generated text talks about shopping in general. "
    "<b>Caveat:</b> words like <i>магазин, отзыв, интернет, покупка</i> appeared in our generation prompts, so part of the signal is the model echoing the prompt.", B)]
rows = [["Trained on", "Model", "F1 same gen.", "F1 unseen gen.", "Unseen recall"]]
for _, r in ru_loo.iterrows():
    rows.append([r.train_on, r.model, f"{r.seen_f1:.3f}", f"{r.unseen_f1:.3f}", f"{r.unseen_fake_recall:.2f}"])
P += [KeepTogether([Paragraph("<b>Generalization to an unseen generator.</b>", B), tbl(rows, [2.4*cm, 2.4*cm, 3.4*cm, 3.6*cm, 3.6*cm]), Spacer(1, 4)]),
  Paragraph(
    "A detector trained on ruGPT-3 transfers well to Qwen. Trained only on Qwen, it misses about a quarter of ruGPT-3 fakes. "
    "That direction is confounded, though: the Qwen training set is about 3× smaller.", B)]
P += [KeepTogether([Paragraph("<b>Held-out modern LLM test (Claude-written reviews).</b>", B), Image(RR + "fig_heldout_claude.png", width=14*cm, height=6.8*cm)]),
  Paragraph(
    f"This is the key result. Trained on ruGPT-3 + Qwen, the best n-gram models catch only "
    f"{ru_ho[ru_ho.model!='stylometric'].claude_recall.min()*100:.0f}–{ru_ho[ru_ho.model!='stylometric'].claude_recall.max()*100:.0f}% of Claude-written fakes "
    f"(combined, controlled: {hg('controlled','combined','claude_recall')*100:.0f}%), while still keeping "
    f"about 95% of real reviews. Raw stylometry catches more ({hg('raw','stylometric','claude_recall')*100:.0f}%) only because Claude's text is always "
    "capitalized and punctuated. That signal is trivial to fake, and it also flags many real reviews "
    f"(specificity {hg('raw','stylometric','real_specificity'):.2f}). <b>Detectors trained on weak generators do not protect against strong ones.</b> "
    "Caveats: the Claude set is small (162 reviews), written in one session (one style), and shorter than average (10 words).", B)]

sec6 = []
sec6 += [PageBreak(), Paragraph("6. Final evaluation protocol and results", H2), Paragraph(
    "<b>Protocol.</b> (1) A fixed, stratified 80/20 train/test split (seed 42); the test split is never used for tuning or selection. "
    "(2) The regularization strength C is chosen by stratified cross-validation <b>on the training split only</b> (EN 3-fold, RU 5-fold), with all "
    "vectorizers refit inside every fold (scikit-learn Pipeline), so validation folds never leak into features. (3) The best configuration is refit "
    "on the full training split, saved as the checkpoint used by the demo, and evaluated <b>once</b> on the test split. "
    "The English test split is identical to the one used in Sections 3–4. For Russian, the split is stratified by source (real / ruGPT-3 / Qwen).", B)]
cfg = [["", "English", "Russian"],
       ["Train / test", f"{DE['split']['train']:,} / {DE['split']['test']:,}", f"{DR['split']['train']:,} / {DR['split']['test']:,}"],
       ["Features", "word 1–2 + char 2–5 TF-IDF", "lemma 1–2 + char 2–5 TF-IDF"],
       ["Classifier / loss", "LinearSVC, L2, squared hinge", "LinearSVC, L2, squared hinge"],
       ["C (CV-selected)", f"{DE['tuning']['proposed_svm']['best_C']} (CV F1 {DE['tuning']['proposed_svm']['cv_f1']:.3f})", f"{DR['tuning']['proposed_svm']['best_C']} (CV F1 {DR['tuning']['proposed_svm']['cv_f1']:.3f})"],
       ["Trainable parameters", f"{DE['model']['trainable_parameters']:,}", f"{DR['model']['trainable_parameters']:,}"],
       ["Fit time (full train)", f"{DE['model']['fit_seconds_full_train']:.1f} s", f"{DR['model']['fit_seconds_full_train']:.1f} s"],
       ["Latency (1 review, median)", f"{DE['timing']['latency_ms_per_review_median']:.1f} ms", f"{DR['timing']['latency_ms_per_review_median']:.1f} ms"],
       ["Checkpoint size", f"{DE['timing']['model_file_mb']:.1f} MB", f"{DR['timing']['model_file_mb']:.1f} MB"]]
sec6 += [tbl(cfg, [4.6*cm, 5.8*cm, 5.8*cm]), Paragraph("Hardware: 2-vCPU Intel Xeon 2.1 GHz, no GPU; scikit-learn 1.9.1; seed 42.", SM), Spacer(1, 6)]
rows = [["Model", "Acc", "Prec", "Rec", "F1", "AUC", "n"]]
for lang, d in [("EN", de), ("RU", dr)]:
    for _, r in d.iterrows():
        rows.append([f"{lang} · " + r.model.replace("PROPOSED: ", "► "), f"{r.accuracy:.3f}", f"{r.precision_macro:.3f}", f"{r.recall_macro:.3f}", f"{r.f1_macro:.3f}", f"{r.roc_auc:.3f}", f"{int(r.test_n):,}"])
sec6 += [Paragraph("<b>Held-out test results</b> (macro-averaged; ► = proposed model):", B), tbl(rows, [7.6*cm, 1.4*cm, 1.4*cm, 1.4*cm, 1.4*cm, 1.4*cm, 1.4*cm]), Spacer(1, 4),
    Paragraph(f"Per class (proposed): EN real P {de.precision_real.iloc[-1]:.3f} / R {de.recall_real.iloc[-1]:.3f}, fake P {de.precision_fake.iloc[-1]:.3f} / R {de.recall_fake.iloc[-1]:.3f}; "
              f"RU real P {dr.precision_real.iloc[-1]:.3f} / R {dr.recall_real.iloc[-1]:.3f}, fake P {dr.precision_fake.iloc[-1]:.3f} / R {dr.recall_fake.iloc[-1]:.3f}. "
              f"Confusion matrices [[TN, FP], [FN, TP]]: EN {de.iloc[-1]['confusion_matrix [[TN,FP],[FN,TP]]']}, RU {dr.iloc[-1]['confusion_matrix [[TN,FP],[FN,TP]]']}. "
              f"RU recall on test by source: ruGPT-3 {DR['recall_by_generator_on_test']['rugpt3']:.3f}, Qwen {DR['recall_by_generator_on_test']['qwen']:.3f}. "
              f"Modern-LLM probe (162 fakes + 162 real, never seen): {DR['modern_llm_probe']['fake_recall']*100:.0f}% of fakes caught, "
              f"{DR['modern_llm_probe']['real_specificity']*100:.0f}% of real reviews kept.", B),
    KeepTogether([Image("results/defense_curves_en.png", width=16*cm, height=6.1*cm), Image("results_ru/defense_curves_ru.png", width=16*cm, height=6.1*cm)]),
    Paragraph("<b>Learning and validation curves.</b> Training F1 stays near 1.0 while validation F1 rises with more data "
              f"(EN {DE['learning_curve']['val_f1'][0]:.2f} → {DE['learning_curve']['val_f1'][-1]:.2f}; RU {DR['learning_curve']['val_f1'][0]:.2f} → {DR['learning_curve']['val_f1'][-1]:.2f}): "
              "a variance gap. The models fit the training data almost perfectly and would still gain from more data. The validation curve over C "
              "shows where stronger regularization stops helping; we pick C at the validation peak.", B),
    KeepTogether([Image("results/defense_epochs_en.png", width=16*cm, height=5.7*cm), Image("results_ru/defense_epochs_ru.png", width=16*cm, height=5.7*cm)]),
    Paragraph("<b>Why no epochs for the SVM?</b> LinearSVC and logistic regression solve a convex problem to convergence with a deterministic solver, "
              "so there is no epoch loop, learning rate or batch size. To show iterative training dynamics, we trained an equivalent linear model "
              "with SGD (log loss, constant learning rate 0.3, L2 α = 1e-6, mini-batch 256) on a train/validation split of the training data. "
              f"Validation loss falls steadily and flattens (EN {DE['sgd_epochs'][0]['val_loss']:.3f} → {DE['sgd_epochs'][-1]['val_loss']:.3f} after 25 epochs) while "
              "training loss keeps falling: the widening gap is mild overfitting, which early stopping or stronger L2 would limit.", B),
    Paragraph(f"<b>Leakage check.</b> For every test review we computed its maximum cosine similarity (word TF-IDF) to any training review. "
              f"English: {DE['near_duplicates']['test_reviews_with_cosine']['>= 0.9']} of {DE['split']['test']:,} test reviews ≥ 0.9 "
              f"(median {DE['near_duplicates']['max_cosine_median']:.2f}); removing them leaves F1 at {DE['near_duplicates']['f1_macro_without_near_dups(>=0.9)']:.3f}. "
              f"Russian: {DR['near_duplicates']['test_reviews_with_cosine']['>= 0.8']} test reviews ≥ 0.8.", B),
    Image("results/defense_cm_en.png", width=6*cm, height=5.3*cm)]
P += sec6

P += [Paragraph("7. Limitations and threats to validity", H2), Paragraph(
    "• The fakes come from <b>GPT-2 (2019)</b>. Reviews from modern LLMs are far more fluent and would very likely defeat these "
    "features. Treat the scores as an upper bound on how easy the task is.<br/>"
    "• <b>Machine-generated ≠ fraudulent.</b> Real fake reviews are often written by paid humans, and none are in this dataset.<br/>"
    "• Russian fakes were generated by us with small models and our own prompts (prompt echo, ruGPT-3 often ignores the requested tone). One domain (clothing). Kaspi/Wildberries reviews are untested.<br/>"
    "• One fixed test split per language; run-to-run variation is reported only as CV standard deviation on the training split (EN ±0.002, RU ±0.009). The Russian test set is small (409).<br/>• The English logistic-regression baseline picked C at the upper edge of its grid.", B)]

P += [Paragraph("8. Conclusion and future work", H2), Paragraph(
    "Classical NLP detects fakes from weak generators very well in both languages (F1 ≈ 0.96), but the scores are inflated by "
    "shortcuts (truncation in English; casing, punctuation and prompt echo in Russian), and they collapse against a modern LLM. "
    "A model trained on one language is useless on the other, so we ship a bilingual detector that routes by script. Next steps: "
    "(1) train on fakes from several strong LLMs with varied prompts; (2) add multilingual transformers (e.g. ruBERT, XLM-R); "
    "(3) test on real Kaspi/Wildberries reviews; (4) add human-written (paid) fake reviews.", B)]

P += [Paragraph("Reproducibility", H2), Paragraph(
    "python defense_eval.py ru / en (≈1 / ≈20 min) → reported results, curves, checkpoints; "
    "python experiments.py (≈4 min on CPU) → main results, ablation, n-grams, errors; then python cross_category_fair.py → "
    "size-matched cross-category test; python generate_ru.py rugpt3 800 / qwen 288 → Russian fakes (≈35 min CPU); "
    "python experiments_ru.py → exploratory Russian results; python predict.py \"text\" → bilingual EN/RU prediction; "
    "python make_report.py → this PDF. Seed 42.", SM)]

SimpleDocTemplate("Fake_Review_Detection_Report_EN_RU.pdf", pagesize=A4, leftMargin=2.2*cm, rightMargin=2.2*cm,
                  topMargin=2*cm, bottomMargin=2*cm, title="Detecting Computer-Generated Product Reviews").build(P)
print("ok")
