"""Russian fake-review detection.
Real: RuReviews (women's clothing & accessories, Smetanin & Komarov 2019).
Fake: ruGPT-3 small (GPT-2-era), Qwen2.5-0.5B-Instruct (modern small LLM) -> train/test;
      Claude-written reviews -> held-out only (strong modern LLM, never seen in training).
Run: python experiments_ru.py
"""
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

def vec(kind):
    if kind == "word": return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    if kind == "char": return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)

def fit_predict(model, tr, te, controlled):
    txt = "text_ctrl" if controlled else "review"
    if model == "stylometric":
        S = lambda d: pd.DataFrame([stylo(t, controlled) for t in d[txt]])
        sc = StandardScaler(); Xtr, Xte = sc.fit_transform(S(tr)), sc.transform(S(te)); clf = LogisticRegression(max_iter=3000)
    elif model == "word": v = vec("word"); Xtr, Xte = v.fit_transform(tr[txt]), v.transform(te[txt]); clf = LogisticRegression(max_iter=3000, C=4)
    elif model == "lemma": v = vec("word"); Xtr, Xte = v.fit_transform(tr.lemmas), v.transform(te.lemmas); clf = LogisticRegression(max_iter=3000, C=4)
    elif model == "char": v = vec("char"); Xtr, Xte = v.fit_transform(tr[txt]), v.transform(te[txt]); clf = LogisticRegression(max_iter=3000, C=4)
    elif model == "combined":
        vl, vc = vec("word"), vec("char"); sc = StandardScaler()
        S = lambda d: pd.DataFrame([stylo(t, controlled) for t in d[txt]])
        Xtr = hstack([vl.fit_transform(tr.lemmas), vc.fit_transform(tr[txt]), csr_matrix(sc.fit_transform(S(tr)))]).tocsr()
        Xte = hstack([vl.transform(te.lemmas), vc.transform(te[txt]), csr_matrix(sc.transform(S(te)))]).tocsr()
        clf = LinearSVC(C=0.5)
    clf.fit(Xtr, tr.y); return clf.predict(Xte), clf.decision_function(Xte)

def metrics(y, p, s):
    return dict(macro_f1=f1_score(y, p, average="macro"), accuracy=accuracy_score(y, p),
                roc_auc=roc_auc_score(y, s) if len(set(y)) > 1 else np.nan)

MODELS = ["stylometric", "word", "lemma", "char", "combined"]
NAMES = {"stylometric": "Stylometric + LR", "word": "Word TF-IDF + LR", "lemma": "Lemma TF-IDF + LR (pymorphy3)",
         "char": "Char TF-IDF + LR", "combined": "Combined (lemma+char+stylo) + SVM"}

# ---------------- E1: 5-fold CV, raw vs controlled ----------------
skf = StratifiedKFold(5, shuffle=True, random_state=SEED); rows, per_gen = [], []
strat = df.generator  # stratify by generator so each fold has all three sources
for controlled in (False, True):
    for m in MODELS:
        fold = []
        for k, (a, b) in enumerate(skf.split(df, strat)):
            tr, te = df.iloc[a], df.iloc[b]; p, s = fit_predict(m, tr, te, controlled); fold.append(metrics(te.y, p, s))
            if controlled and m == "combined":
                for g in ["rugpt3", "qwen"]: per_gen.append(dict(generator=g, recall=(p[(te.generator == g).values] == 1).mean()))
                per_gen.append(dict(generator="real (specificity)", recall=(p[(te.generator == "real").values] == 0).mean()))
        f = pd.DataFrame(fold)
        rows.append(dict(setting="controlled" if controlled else "raw", model=m,
                         **{f"{c}_mean": f[c].mean() for c in f}, **{f"{c}_std": f[c].std() for c in f}))
        print(rows[-1]["setting"], m, round(rows[-1]["macro_f1_mean"], 3), "±", round(rows[-1]["macro_f1_std"], 3))
main = pd.DataFrame(rows); main.to_csv(OUT / "main_results.csv", index=False)
per_gen = pd.DataFrame(per_gen).groupby("generator").recall.mean(); per_gen.to_csv(OUT / "per_generator_recall.csv")

# ---------------- E2: leave-one-generator-out (controlled) ----------------
loo = []
real_tr, real_te = train_test_split(df[df.y == 0], test_size=0.3, random_state=SEED)
for train_g, test_g in [("rugpt3", "qwen"), ("qwen", "rugpt3")]:
    gtr = df[df.generator == train_g]; gte = df[df.generator == test_g]
    rtr = real_tr.sample(min(len(gtr), len(real_tr)), random_state=SEED)
    rte = real_te.sample(min(len(gte), len(real_te)), random_state=SEED)
    # in-distribution reference: held-out part of the same generator
    g_a, g_b = train_test_split(gtr, test_size=0.3, random_state=SEED)
    for m in ["lemma", "char", "combined"]:
        tr = pd.concat([g_a, rtr.iloc[:len(g_a)]]); te_seen = pd.concat([g_b, rte.iloc[:len(g_b)]]); te_unseen = pd.concat([gte, rte])
        p1, s1 = fit_predict(m, tr, te_seen, True); p2, s2 = fit_predict(m, tr, te_unseen, True)
        loo.append(dict(train_on=train_g, model=m, seen_f1=metrics(te_seen.y, p1, s1)["macro_f1"],
                        unseen_f1=metrics(te_unseen.y, p2, s2)["macro_f1"],
                        unseen_fake_recall=(p2[te_unseen.y.values == 1] == 1).mean()))
        print(loo[-1])
loo = pd.DataFrame(loo); loo.to_csv(OUT / "leave_one_generator_out.csv", index=False)

# ---------------- E3: held-out Claude-written fakes ----------------
claude_rows = []
for controlled in (False, True):
    for m in MODELS:
        p, s = fit_predict(m, df, ho, controlled)
        claude_rows.append(dict(setting="controlled" if controlled else "raw", model=m, **metrics(ho.y, p, s),
                                claude_recall=(p[ho.y.values == 1] == 1).mean(), real_specificity=(p[ho.y.values == 0] == 0).mean()))
claude_res = pd.DataFrame(claude_rows); claude_res.to_csv(OUT / "heldout_claude.csv", index=False)
print(claude_res.round(3).to_string())

# ---------------- E4: does the English model work on Russian? ----------------
en = pd.read_csv("data/fake.csv").dropna(subset=["text_"]); en["y"] = (en.label == "CG").astype(int)
ven = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True); clf_en = LogisticRegression(max_iter=3000, C=4).fit(ven.fit_transform(en.text_), en.y)
Xru = ven.transform(df.review); p_en = clf_en.predict(Xru)
en_on_ru = dict(macro_f1=f1_score(df.y, p_en, average="macro"), roc_auc=roc_auc_score(df.y, clf_en.decision_function(Xru)),
                share_all_zero_rows=float((Xru.getnnz(axis=1) == 0).mean()))
print("EN model on RU:", en_on_ru)

# ---------------- top lemmas & stats ----------------
v = vec("word"); X = v.fit_transform(df.lemmas); clf = LogisticRegression(max_iter=3000, C=4).fit(X, df.y)
voc = np.array(v.get_feature_names_out()); o = np.argsort(clf.coef_[0])
top = {"fake": voc[o[-20:]][::-1].tolist(), "real": voc[o[:20]].tolist()}
stats = {g: dict(n=int(len(d)), mean_words=float(d.review.str.split().str.len().mean()),
                 ends_punct=float(d.review.str.strip().str[-1].isin(list(".!?)")).mean()),
                 starts_upper=float(d.review.str[:1].str.isupper().mean()))
         for g, d in pd.concat([df, ho]).groupby("generator")}
json.dump(dict(n_raw_generated=n_raw, n_kept_after_filter=n_kept, stats=stats, top_lemmas=top, en_model_on_ru=en_on_ru),
          open(OUT / "summary.json", "w"), indent=1, ensure_ascii=False)
ex = df[df.y == 1].groupby("generator").sample(4, random_state=1)[["generator", "review"]]
ex = pd.concat([ex, claude.sample(4, random_state=1)[["generator", "review"]], real.sample(4, random_state=1)[["generator", "review"]]])
ex.to_csv(OUT / "examples.csv", index=False)

# ---------------- charts ----------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
pv = main.pivot(index="model", columns="setting", values="macro_f1_mean").loc[MODELS]
er = main.pivot(index="model", columns="setting", values="macro_f1_std").loc[MODELS]
fig, ax = plt.subplots(figsize=(7, 3.6)); x = np.arange(len(MODELS)); w = .38
ax.bar(x - w/2, pv.raw, w, yerr=er.raw, capsize=3, label="raw text", color="#4C6EF5")
ax.bar(x + w/2, pv.controlled, w, yerr=er.controlled, capsize=3, label="controlled (lowercase, no punctuation)", color="#F08C00")
ax.set_xticks(x, ["Stylometric", "Word", "Lemma", "Char", "Combined"]); ax.set_ylim(0.5, 1); ax.set_ylabel("Macro-F1 (5-fold CV)")
ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2); fig.tight_layout(); fig.savefig(OUT / "fig_models_ru.png", dpi=200)

c = claude_res[claude_res.setting == "controlled"].set_index("model").loc[MODELS]
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(x - w/2, c.claude_recall, w, label="Claude fakes caught (recall)", color="#E03131")
ax.bar(x + w/2, c.real_specificity, w, label="real reviews kept (specificity)", color="#ADB5BD")
ax.set_xticks(x, ["Stylometric", "Word", "Lemma", "Char", "Combined"]); ax.set_ylim(0, 1.05); ax.set_ylabel("Share")
ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2); fig.tight_layout(); fig.savefig(OUT / "fig_heldout_claude.png", dpi=200)

# The deployed/reported Russian checkpoint (results_ru/model_ru.joblib) is produced by defense_eval.py
# (tuned on the train split only, evaluated once on the held-out test split). This script is exploratory.
print("done")
