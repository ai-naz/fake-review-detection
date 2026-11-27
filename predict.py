"""Bilingual (EN/RU) fake-review detector.
  python predict.py "Отличное платье, очень довольна!" "Great product, works as described."
  python predict.py --file reviews.txt        (one review per line)
Language is detected by script (Cyrillic share). RU model: results_ru/model_ru.joblib (from experiments_ru.py).
EN model is trained on first use from data/fake.csv and cached in results/model_en.joblib.
"""
import sys, re, string
from pathlib import Path
import joblib, pandas as pd
from scipy.sparse import hstack

def is_russian(t):
    letters = [c for c in t if c.isalpha()]
    return bool(letters) and sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in letters) / len(letters) >= 0.5

def control_ru(t): return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()
def control_en(t): return re.sub(r"\s+", " ", t).strip().rstrip(string.punctuation + " ")

_morph, _cache = None, {}
def lemmatize(t):
    global _morph
    if _morph is None:
        import pymorphy3; _morph = pymorphy3.MorphAnalyzer()
    out = []
    for w in re.findall(r"[а-яёa-z0-9]+", t.lower()):
        if w not in _cache: _cache[w] = _morph.parse(w)[0].normal_form
        out.append(_cache[w])
    return " ".join(out)

EN_C = 1  # chosen by 3-fold CV on the train split (results/defense_report_en.json)


def load_en():
    """Load the reported English checkpoint; if missing, rebuild it exactly as defense_eval.py does
    (train split only: 80%, stratified, seed 42; word+char TF-IDF + LinearSVC with the CV-tuned C)."""
    p = Path("results/model_en.joblib")
    if p.exists(): return joblib.load(p)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    print("Building English checkpoint from the train split (first run, ~1-2 min)...", file=sys.stderr)
    import experiments_core as E
    tr, _ = train_test_split(E.df, test_size=0.2, stratify=E.df.y, random_state=42)
    t = tr.text_norm.to_numpy(dtype=object)
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True)
    X = hstack([vw.fit_transform(t), vc.fit_transform(t)]).tocsr()
    m = dict(vw=vw, vc=vc, clf=LinearSVC(C=EN_C, random_state=42).fit(X, tr.y.to_numpy()), trained_on="train split (80%, seed 42)", C=EN_C)
    p.parent.mkdir(exist_ok=True); joblib.dump(m, p); return m


def load_ru():
    p = Path("results_ru/model_ru.joblib")
    if not p.exists():
        sys.exit("Russian checkpoint missing: run  python defense_eval.py ru  (about 1 minute) first.")
    return joblib.load(p)


def explain(text, k=6):
    """Preprocessing steps and the k features that pushed the score most (for the demo)."""
    import numpy as np
    if is_russian(text):
        m, c = load_ru(), control_ru(text); lem = lemmatize(c)
        names = [f"lemma: {w}" for w in m["vl"].get_feature_names_out()] + [f"chars: '{w}'" for w in m["vc"].get_feature_names_out()]
        X = hstack([m["vl"].transform([lem]), m["vc"].transform([c])]).tocsr()
        steps = {"language": "Russian", "normalized": c, "lemmas": lem, "word tokens": m["vl"].build_analyzer()(lem)[:20]}
    else:
        m, c = load_en(), control_en(text)
        names = [f"word: {w}" for w in m["vw"].get_feature_names_out()] + [f"chars: '{w}'" for w in m["vc"].get_feature_names_out()]
        X = hstack([m["vw"].transform([c]), m["vc"].transform([c])]).tocsr()
        steps = {"language": "English", "normalized": c, "word tokens": m["vw"].build_analyzer()(c)[:20]}
    w = m["clf"].coef_.ravel(); contrib = X.multiply(w).tocsr()
    idx, val = contrib.indices, contrib.data
    order = np.argsort(val)
    steps["non-zero features"] = int(X.nnz)
    steps["toward FAKE"] = [(names[idx[i]], round(float(val[i]), 3)) for i in order[::-1][:k] if val[i] > 0]
    steps["toward REAL"] = [(names[idx[i]], round(float(val[i]), 3)) for i in order[:k] if val[i] < 0]
    steps["score"] = float(m["clf"].decision_function(X)[0])
    return steps


def predict(texts):
    ru_m, en_m, res = None, None, []
    for t in texts:
        if is_russian(t):
            ru_m = ru_m or load_ru()
            c = control_ru(t); X = hstack([ru_m["vl"].transform([lemmatize(c)]), ru_m["vc"].transform([c])]); s = ru_m["clf"].decision_function(X)[0]; lang = "ru"
        else:
            en_m = en_m or load_en()
            c = control_en(t); X = hstack([en_m["vw"].transform([c]), en_m["vc"].transform([c])]); s = en_m["clf"].decision_function(X)[0]; lang = "en"
        res.append(dict(lang=lang, label="FAKE (machine-generated)" if s > 0 else "REAL", score=round(float(s), 3), text=t))
    return res

if __name__ == "__main__":
    a = sys.argv[1:]
    texts = [l.strip() for l in open(a[1], encoding="utf-8") if l.strip()] if a[:1] == ["--file"] else a
    for r in predict(texts):
        print(f"[{r['lang']}] {r['label']:<25} score={r['score']:+.3f}  {r['text'][:80]}")
