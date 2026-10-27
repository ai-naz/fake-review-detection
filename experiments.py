"""Fake (computer-generated) review detection — classical NLP experiments.
Dataset: Salminen et al. (2022) fake reviews dataset (CG = GPT-2 generated, OR = original Amazon).
Run: python experiments.py  -> writes results/*.json, *.csv, *.png
"""
import json, re, string
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix

SEED = 42
OUT = Path("results"); OUT.mkdir(exist_ok=True)

# ---------- data ----------
df = pd.read_csv("data/fake.csv").rename(columns={"text_": "text"})
df = df.dropna(subset=["text"]).drop_duplicates("text").reset_index(drop=True)
df["y"] = (df.label == "CG").astype(int)          # 1 = fake
df["category"] = df.category.str.replace("_5", "", regex=False)

def normalize(t):
    """Artifact-controlled text: collapse whitespace, drop trailing punctuation."""
    t = re.sub(r"\s+", " ", t).strip()
    return t.rstrip(string.punctuation + " ")
df["text_norm"] = df.text.map(normalize)

# ---------- stylometric features ----------
FUNC = set("the a an and or but of to in on for with is it this that i my was".split())
def stylo(t, controlled=False):
    w = t.split(); n = max(len(w), 1)
    sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    f = {
        "n_words": len(w),
        "avg_word_len": np.mean([len(x) for x in w]) if w else 0,
        "type_token_ratio": len(set(x.lower() for x in w)) / n,
        "n_sentences": len(sents),
        "avg_sent_len": n / max(len(sents), 1),
        "punct_ratio": sum(c in string.punctuation for c in t) / max(len(t), 1),
        "exclam": t.count("!") / n,
        "upper_ratio": sum(c.isupper() for c in t) / max(len(t), 1),
        "first_person": sum(x.lower() in {"i", "me", "my", "mine"} for x in w) / n,
        "func_word_ratio": sum(x.lower() in FUNC for x in w) / n,
        "digit_ratio": sum(c.isdigit() for c in t) / max(len(t), 1),
    }
    if not controlled:
        f["ends_with_punct"] = int(t.strip()[-1:] in ".!?)") if t.strip() else 0
        f["double_space"] = int("  " in t)
    return f

# ---------- models ----------
def build(kind, tr_txt, te_txt, tr_df, te_df, controlled):
    if kind == "stylometric":
        S = lambda s: pd.DataFrame([stylo(x, controlled) for x in s])
        sc = StandardScaler(); Xtr = sc.fit_transform(S(tr_txt)); Xte = sc.transform(S(te_txt))
        return Xtr, Xte, LogisticRegression(max_iter=2000)
    if kind.startswith("tfidf_word"):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    elif kind.startswith("tfidf_char"):
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True)
    if kind == "combined":
        vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True)
        S = lambda s: pd.DataFrame([stylo(x, controlled) for x in s])
        sc = StandardScaler()
        Xtr = hstack([vw.fit_transform(tr_txt), vc.fit_transform(tr_txt), csr_matrix(sc.fit_transform(S(tr_txt)))]).tocsr()
        Xte = hstack([vw.transform(te_txt), vc.transform(te_txt), csr_matrix(sc.transform(S(te_txt)))]).tocsr()
        return Xtr, Xte, LinearSVC(C=0.5)
    Xtr = v.fit_transform(tr_txt); Xte = v.transform(te_txt)
    clf = LinearSVC(C=0.5) if kind.endswith("svm") else LogisticRegression(max_iter=2000, C=4)
    return Xtr, Xte, clf, v

def evaluate(kind, tr, te, controlled=False):
    col = "text_norm" if controlled else "text"
    res = build(kind, tr[col], te[col], tr, te, controlled)
    Xtr, Xte, clf = res[:3]
    clf.fit(Xtr, tr.y)
    pred = clf.predict(Xte)
    score = clf.decision_function(Xte)
    m = dict(accuracy=accuracy_score(te.y, pred), macro_f1=f1_score(te.y, pred, average="macro"),
             roc_auc=roc_auc_score(te.y, score))
    return m, clf, res, pred

tr, te = train_test_split(df, test_size=0.2, stratify=df.y, random_state=SEED)
MODELS = ["stylometric", "tfidf_word_lr", "tfidf_char_lr", "tfidf_word_svm", "combined"]

rows = []
for controlled in [False, True]:
    for k in MODELS:
        m, *_ = evaluate(k, tr, te, controlled)
        rows.append(dict(setting="artifact-controlled" if controlled else "raw", model=k, **m))
        print(rows[-1])
main = pd.DataFrame(rows); main.to_csv(OUT / "main_results.csv", index=False)

# ---------- stylometric ablation (raw): drop one feature at a time ----------
abl = []
base, *_ = evaluate("stylometric", tr, te)
feat_names = list(stylo("x y. z").keys())
S_tr = pd.DataFrame([stylo(x) for x in tr.text]); S_te = pd.DataFrame([stylo(x) for x in te.text])
for f in feat_names:
    sc = StandardScaler(); a = sc.fit_transform(S_tr.drop(columns=f)); b = sc.transform(S_te.drop(columns=f))
    clf = LogisticRegression(max_iter=2000).fit(a, tr.y)
    abl.append(dict(dropped=f, macro_f1=f1_score(te.y, clf.predict(b), average="macro"), delta=f1_score(te.y, clf.predict(b), average="macro") - base["macro_f1"]))
abl = pd.DataFrame(abl).sort_values("delta"); abl.to_csv(OUT / "stylometric_ablation.csv", index=False)
sc = StandardScaler(); clf = LogisticRegression(max_iter=2000).fit(sc.fit_transform(S_tr), tr.y)
coef = pd.Series(clf.coef_[0], index=feat_names).sort_values(); coef.to_csv(OUT / "stylometric_coefs.csv")

# ---------- top n-grams (controlled, word LR) ----------
_, clf, res, pred_ctrl = evaluate("tfidf_word_lr", tr, te, controlled=True)
vocab = np.array(res[3].get_feature_names_out()); c = clf.coef_[0]; o = np.argsort(c)
json.dump({"fake": vocab[o[-25:]][::-1].tolist(), "real": vocab[o[:25]].tolist()}, open(OUT / "top_ngrams.json", "w"), indent=1)

# ---------- error analysis sample ----------
te2 = te.assign(pred=pred_ctrl)
err = te2[te2.pred != te2.y]
err.assign(error=np.where(err.y == 1, "fake→predicted real", "real→predicted fake"))[["category", "rating", "label", "error", "text"]] \
   .sample(min(40, len(err)), random_state=SEED).to_csv(OUT / "errors_sample.csv", index=False)
cm = confusion_matrix(te.y, pred_ctrl).tolist()

# ---------- per-rating & length-bucket F1 (controlled word LR) ----------
te2["len_bucket"] = pd.cut(te2.text.str.split().str.len(), [0, 20, 50, 100, 1000], labels=["≤20", "21-50", "51-100", ">100"])
by_len = te2.groupby("len_bucket", observed=True).apply(lambda g: f1_score(g.y, g.pred, average="macro")).rename("macro_f1")
by_len.to_csv(OUT / "f1_by_length.csv")

# ---------- leave-one-category-out (controlled, word LR) ----------
loco = []
for cat in sorted(df.category.unique()):
    a, b = df[df.category != cat], df[df.category == cat]
    m, *_ = evaluate("tfidf_word_lr", a, b, controlled=True)
    # in-domain reference: same category, 80/20 split
    a2, b2 = train_test_split(b, test_size=0.2, stratify=b.y, random_state=SEED)
    m2, *_ = evaluate("tfidf_word_lr", a2, b2, controlled=True)
    loco.append(dict(held_out=cat, cross_f1=m["macro_f1"], in_domain_f1=m2["macro_f1"], n=len(b)))
    print(loco[-1])
loco = pd.DataFrame(loco); loco.to_csv(OUT / "cross_category.csv", index=False)

json.dump(dict(n_reviews=len(df), train=len(tr), test=len(te), confusion_matrix_controlled_word_lr=cm,
               end_punct_rate={l: float(df[df.label == l].text.str.strip().str[-1].isin(list(".!?)")).mean()) for l in ["CG", "OR"]},
               mean_words={l: float(df[df.label == l].text.str.split().str.len().mean()) for l in ["CG", "OR"]}),
          open(OUT / "summary.json", "w"), indent=1)

# ---------- charts ----------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
p = main.pivot(index="model", columns="setting", values="macro_f1").loc[MODELS]
fig, ax = plt.subplots(figsize=(7, 3.6)); x = np.arange(len(p)); w = 0.38
ax.bar(x - w/2, p["raw"], w, label="raw text", color="#4C6EF5"); ax.bar(x + w/2, p["artifact-controlled"], w, label="artifact-controlled", color="#F08C00")
ax.set_xticks(x, ["Stylometric\nLR", "Word TF-IDF\nLR", "Char TF-IDF\nLR", "Word TF-IDF\nSVM", "Combined\nSVM"]); ax.set_ylim(0.5, 1); ax.set_ylabel("Macro-F1"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig_models.png", dpi=200)

fig, ax = plt.subplots(figsize=(7, 3.6)); l = loco.sort_values("cross_f1"); y = np.arange(len(l))
ax.barh(y - 0.2, l.in_domain_f1, 0.4, label="in-domain (same category)", color="#ADB5BD"); ax.barh(y + 0.2, l.cross_f1, 0.4, label="cross-category (unseen)", color="#4C6EF5")
ax.set_yticks(y, l.held_out.str.replace("_", " ")); ax.set_xlim(0.5, 1); ax.set_xlabel("Macro-F1"); ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig_cross_category.png", dpi=200)
print("done")
