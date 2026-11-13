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

def load_en():
    p = Path("results/model_en.joblib")
    if p.exists(): return joblib.load(p)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    print("Training English model (first run, ~1-2 min)...", file=sys.stderr)
    d = pd.read_csv("data/fake.csv").dropna(subset=["text_"]); d["t"] = d.text_.map(control_en)
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True)
    X = hstack([vw.fit_transform(d.t), vc.fit_transform(d.t)]).tocsr()
    m = dict(vw=vw, vc=vc, clf=LinearSVC(C=0.5).fit(X, (d.label == "CG").astype(int)))
    p.parent.mkdir(exist_ok=True); joblib.dump(m, p); return m

def predict(texts):
    ru_m, en_m, res = None, None, []
    for t in texts:
        if is_russian(t):
            ru_m = ru_m or joblib.load("results_ru/model_ru.joblib")
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
