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

