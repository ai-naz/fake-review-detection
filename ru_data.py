"""Russian data loading + preprocessing, shared with defense_eval.py (identical to the top of experiments_ru.py)."""
import json, re, string
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pymorphy3, joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score

SEED = 42; OUT = Path("results_ru"); OUT.mkdir(exist_ok=True)
morph = pymorphy3.MorphAnalyzer()

# ---------------- data ----------------
def cyr_ok(t):
    letters = [c for c in t if c.isalpha()]
    return len(letters) > 0 and sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in letters) / len(letters) >= 0.8

fake = pd.concat([pd.read_csv("data_ru/fake_rugpt3.csv"), pd.read_csv("data_ru/fake_qwen.csv")]).dropna(subset=["review"])
n_raw = fake.groupby("generator").size().to_dict()
META = r"(?i)извин|прошу прощени|я не являюсь|я создан|как (ии|искусственный)|попробуем создать|вот (его|ваш|пример)|напиши|отзыв покупател"
fake = fake[~fake.review.str.contains(META)]
fake = fake[fake.review.map(cyr_ok) & (fake.review.str.split().str.len() >= 3)].drop_duplicates("review")
n_kept = fake.groupby("generator").size().to_dict()
claude = pd.read_csv("data_ru/fake_claude.txt", sep="|", names=["sentiment", "review"]).assign(generator="claude")

real_all = pd.read_csv("data_ru/rureviews.csv", sep="\t").dropna()
real_all = real_all[real_all.review.str.split().str.len().between(3, 80) & real_all.review.map(cyr_ok)].drop_duplicates("review")
# real sample matched to fakes on sentiment; separate disjoint pool for the held-out Claude test
real = pd.concat([real_all[real_all.sentiment == s].sample(k, random_state=SEED)
                  for s, k in fake.sentiment.value_counts().items()]).assign(generator="real")
pool = real_all.drop(real.index)
real_ho = pd.concat([pool[pool.sentiment == s].sample(k, random_state=SEED)
                     for s, k in claude.sentiment.value_counts().items()]).assign(generator="real")

df = pd.concat([fake, real]).reset_index(drop=True); df["y"] = (df.generator != "real").astype(int)
ho = pd.concat([claude, real_ho]).reset_index(drop=True); ho["y"] = (ho.generator != "real").astype(int)

# artifact control: real reviews are often lowercase/unpunctuated, generated ones are tidy -> strip both
def control(t): return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()
for d in (df, ho): d["text_ctrl"] = d.review.map(control)

_cache = {}
def lemmatize(t):
    out = []
    for w in re.findall(r"[а-яёa-z0-9]+", t.lower()):
        if w not in _cache: _cache[w] = morph.parse(w)[0].normal_form
        out.append(_cache[w])
    return " ".join(out)
for d in (df, ho): d["lemmas"] = d.text_ctrl.map(lemmatize)

# ---------------- features ----------------
FUNC_RU = set("и в не на что я с а как это но по к у же за из от так все она он мне очень было бы для то".split())
def stylo(t, controlled):
    w = t.split(); n = max(len(w), 1); sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    f = dict(n_words=len(w), avg_word_len=np.mean([len(x) for x in w]) if w else 0,
             ttr=len(set(x.lower() for x in w)) / n, func_ratio=sum(x.lower().strip(string.punctuation) in FUNC_RU for x in w) / n,
             first_person=sum(x.lower().strip(string.punctuation) in {"я", "мне", "меня", "мой", "моя", "мое", "мои"} for x in w) / n)
    if not controlled:
        f.update(n_sent=len(sents), punct_ratio=sum(c in string.punctuation for c in t) / max(len(t), 1),
                 exclam=t.count("!") / n, upper_ratio=sum(c.isupper() for c in t) / max(len(t), 1),
                 starts_upper=int(t[:1].isupper()), ends_punct=int(t.strip()[-1:] in ".!?)"))
    return f

