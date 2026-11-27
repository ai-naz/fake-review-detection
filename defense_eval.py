"""Defense evaluation: the reported, reproducible numbers for both languages.

Protocol (same for EN and RU):
  1. Fixed held-out TEST split (stratified, seed 42). It is never used for tuning or model selection.
  2. Hyperparameters (regularization C) chosen by stratified K-fold cross-validation on the TRAIN split only
     (vectorizers are refit inside every fold, so no information leaks from validation folds).
  3. Best configuration refit on the full TRAIN split -> saved checkpoint -> evaluated ONCE on TEST.
  4. Extra evidence: learning curves, validation (C) curves, an epoch-wise training curve of an equivalent
     iteratively-trained linear model (SGD, log loss), confusion matrices, per-class metrics, timing and size,
     and a near-duplicate leakage check between TRAIN and TEST.

Run:  python defense_eval.py        (~20-30 min on a laptop CPU; writes results/defense_* and results_ru/defense_*)
"""
import json, time, platform, os, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score, roc_auc_score,
                             confusion_matrix, log_loss)

warnings.filterwarnings("ignore")
SEED = 42
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
HW = {"cpu": platform.processor() or platform.machine(), "logical_cores": os.cpu_count(),
      "python": platform.python_version(), "os": platform.system(), "gpu": "none (CPU only)"}


# ---------------------------------------------------------------- helpers
def word_vec(): return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
def char_vec(): return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True)


def scores(model, X):
    return model.decision_function(X) if hasattr(model, "decision_function") else model.predict_proba(X)[:, 1]


def full_metrics(name, y, pred, s=None):
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {"model": name, "accuracy": accuracy_score(y, pred),
            "precision_macro": p.mean(), "recall_macro": r.mean(), "f1_macro": f.mean(),
            "precision_real": p[0], "recall_real": r[0], "f1_real": f[0],
            "precision_fake": p[1], "recall_fake": r[1], "f1_fake": f[1],
            "roc_auc": roc_auc_score(y, s) if s is not None and len(set(s)) > 1 else 0.5,  # constant score = random ranking
            "test_n": int(len(y)), "confusion_matrix [[TN,FP],[FN,TP]]": confusion_matrix(y, pred).tolist()}


def plot_cm(cm, title, path):
    cm = np.array(cm); fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["real", "fake"]); ax.set_yticks([0, 1], ["real", "fake"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_curves(lc, vc, title, path, cname="C"):
    sizes, tr, va = lc
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    ax.plot(sizes, tr.mean(1), "o-", color="#4C6EF5", label="train")
    ax.fill_between(sizes, va.mean(1) - va.std(1), va.mean(1) + va.std(1), color="#F08C00", alpha=.2)
    ax.plot(sizes, va.mean(1), "o-", color="#F08C00", label="validation (CV)")
    ax.set_xlabel("training examples"); ax.set_ylabel("macro-F1"); ax.set_title("Learning curve"); ax.legend(frameon=False)
    ax = axes[1]
    cs, trc, vac = vc
    ax.semilogx(cs, trc, "o-", color="#4C6EF5", label="train")
    ax.semilogx(cs, vac, "o-", color="#F08C00", label="validation (CV)")
    best = cs[int(np.argmax(vac))]; ax.axvline(best, ls="--", color="grey"); ax.text(best, min(vac), f" chosen {cname}={best}", fontsize=8)
    ax.set_xlabel(f"regularization {cname} (higher = less regularization)"); ax.set_ylabel("macro-F1")
    ax.set_title("Validation curve (hyperparameter)"); ax.legend(frameon=False)
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def sgd_epoch_curve(Xtr, ytr, Xva, yva, epochs, path, title):
    """Epoch-wise loss/metric curve for an iteratively trained linear model (logistic loss, SGD)."""
    clf = SGDClassifier(loss="log_loss", alpha=1e-6, learning_rate="constant", eta0=0.3, random_state=SEED)  # settings picked on validation loss
    rng = np.random.RandomState(SEED); rows = []
    for ep in range(1, epochs + 1):
        idx = rng.permutation(Xtr.shape[0])
        for b in range(0, len(idx), 256):          # mini-batch size 256
            j = idx[b:b + 256]; clf.partial_fit(Xtr[j], ytr[j], classes=[0, 1])
        rows.append(dict(epoch=ep, train_loss=log_loss(ytr, clf.predict_proba(Xtr)),
                         val_loss=log_loss(yva, clf.predict_proba(Xva)),
                         val_f1=f1_score(yva, clf.predict(Xva), average="macro")))
    h = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(h.epoch, h.train_loss, "o-", color="#4C6EF5", label="train loss")
    ax[0].plot(h.epoch, h.val_loss, "o-", color="#F08C00", label="validation loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("log loss (cross-entropy)"); ax[0].legend(frameon=False); ax[0].set_title("Loss")
    ax[1].plot(h.epoch, h.val_f1, "o-", color="#2B8A3E"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation macro-F1"); ax[1].set_title("Validation metric")
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return h


def near_duplicates(train_texts, test_texts, thresholds=(0.8, 0.9)):
    """Max cosine similarity of every test review to any train review (word TF-IDF fit on train)."""
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(train_texts)
    A, B = v.transform(test_texts), v.transform(train_texts).T.tocsc()
    mx = np.zeros(A.shape[0])
    for i in range(0, A.shape[0], 1000):
        mx[i:i + 1000] = (A[i:i + 1000] @ B).max(axis=1).toarray().ravel()
    return mx, {f">= {t}": int((mx >= t).sum()) for t in thresholds}


def timing(model, X_one_by_one, X_batch, path_model):
    joblib.dump(model, path_model)
    lat = []
    for x in X_one_by_one[:300]:
        t = time.perf_counter(); model.predict(x); lat.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter(); model.predict(X_batch); dt = time.perf_counter() - t
    return {"latency_ms_per_review_median": float(np.median(lat)), "latency_ms_p95": float(np.percentile(lat, 95)),
            "throughput_reviews_per_s": len(X_batch) / dt, "model_file_mb": os.path.getsize(path_model) / 1e6}


def n_params(pipe):
    clf = pipe.named_steps["clf"]; return int(clf.coef_.size + clf.intercept_.size)


# ================================================================ ENGLISH
def run_en():
    import experiments_core as E                      # same data loading, cleaning and dedup as experiments.py
    out = Path("results"); df = E.df
    tr, te = train_test_split(df, test_size=0.2, stratify=df.y, random_state=SEED)   # identical split to experiments.py
    Xtr, Xte = tr.text_norm.to_numpy(dtype=object), te.text_norm.to_numpy(dtype=object)   # plain arrays (pandas 3 string arrays break sklearn indexing)
    ytr, yte = tr.y.to_numpy(), te.y.to_numpy()
    rep = {"hardware": HW, "split": {"train": len(tr), "test": len(te), "seed": SEED, "stratified": True,
                                     "class_balance_train_fake": float(ytr.mean()), "class_balance_test_fake": float(yte.mean())}}
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)

    # --- model definitions (all on artifact-controlled text, except the style baseline which shows the shortcut)
    word_lr = Pipeline([("tfidf", word_vec()), ("clf", LogisticRegression(max_iter=3000))])
    proposed = Pipeline([("feats", FeatureUnion([("word", word_vec()), ("char", char_vec())])), ("clf", LinearSVC(random_state=SEED))])

    # --- hyperparameter selection on TRAIN only
    t0 = time.time()
    g_lr = GridSearchCV(word_lr, {"clf__C": [0.5, 1, 2, 4, 8, 16]}, scoring="f1_macro", cv=cv, n_jobs=2, return_train_score=True).fit(Xtr, ytr)
    g_sv = GridSearchCV(proposed, {"clf__C": [0.05, 0.1, 0.25, 0.5, 1, 2]}, scoring="f1_macro", cv=cv, n_jobs=2, return_train_score=True).fit(Xtr, ytr)
    rep["tuning"] = {"protocol": "3-fold stratified CV on train only, metric macro-F1",
                     "word_lr": {"grid": [0.5, 1, 2, 4, 8, 16], "best_C": g_lr.best_params_["clf__C"], "cv_f1": g_lr.best_score_},
                     "proposed_svm": {"grid": [0.05, 0.1, 0.25, 0.5, 1, 2], "best_C": g_sv.best_params_["clf__C"], "cv_f1": g_sv.best_score_,
                                      "cv_f1_std": float(g_sv.cv_results_["std_test_score"][g_sv.best_index_])},
                     "tuning_seconds": time.time() - t0}
    print("EN tuning", rep["tuning"], flush=True)

    # --- learning curve for the proposed model (train split only)
    lc = learning_curve(g_sv.best_estimator_, Xtr, ytr, train_sizes=[0.05, 0.1, 0.25, 0.5, 1.0], cv=cv,
                        scoring="f1_macro", n_jobs=2, shuffle=True, random_state=SEED)
    vc = ([0.05, 0.1, 0.25, 0.5, 1, 2], g_sv.cv_results_["mean_train_score"], g_sv.cv_results_["mean_test_score"])
    plot_curves(lc, vc, "English: word+char TF-IDF + linear SVM (train split, 3-fold CV)", out / "defense_curves_en.png")
    rep["learning_curve"] = {"train_sizes": lc[0].tolist(), "train_f1": lc[1].mean(1).tolist(), "val_f1": lc[2].mean(1).tolist()}

    # --- epoch-wise training curve (SGD logistic regression on word TF-IDF; 10% of train as validation)
    a, b = train_test_split(np.arange(len(Xtr)), test_size=0.1, stratify=ytr, random_state=SEED)
    v = word_vec().fit(Xtr[a])
    h = sgd_epoch_curve(v.transform(Xtr[a]), ytr[a], v.transform(Xtr[b]), ytr[b], 25, out / "defense_epochs_en.png",
                        "English: SGD logistic regression, word TF-IDF (train 90% / validation 10% of train split)")
    rep["sgd_epochs"] = h.round(4).to_dict("records")

    # --- final fit on full TRAIN, single evaluation on TEST
    results = []
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    results.append(full_metrics("Majority-class baseline", yte, dummy.predict(Xte), np.zeros(len(yte))))
    S = lambda d: pd.DataFrame([E.stylo(x) for x in d])           # raw text, includes the punctuation shortcut
    sty = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]).fit(S(tr.text), ytr)
    results.append(full_metrics("Stylometric + LR (raw text)", yte, sty.predict(S(te.text)), sty.decision_function(S(te.text))))
    t = time.time(); best_lr = g_lr.best_estimator_.fit(Xtr, ytr); lr_fit = time.time() - t
    results.append(full_metrics(f"Word TF-IDF + LR (C={g_lr.best_params_['clf__C']})", yte, best_lr.predict(Xte), scores(best_lr, Xte)))
    t = time.time(); final = g_sv.best_estimator_.fit(Xtr, ytr); fit_s = time.time() - t
    pred = final.predict(Xte)
    results.append(full_metrics(f"PROPOSED: word+char TF-IDF + linear SVM (C={g_sv.best_params_['clf__C']})", yte, pred, scores(final, Xte)))
    res = pd.DataFrame(results); res.to_csv(out / "defense_results_en.csv", index=False)
    print(res[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "test_n"]].round(4).to_string(), flush=True)
    plot_cm(results[-1]["confusion_matrix [[TN,FP],[FN,TP]]"], "English test set (proposed model)", out / "defense_cm_en.png")

    # --- checkpoint used by predict.py / app.py (trained on TRAIN split only = exactly the reported model)
    vw, vc_ = final.named_steps["feats"].transformer_list[0][1], final.named_steps["feats"].transformer_list[1][1]
    joblib.dump(dict(vw=vw, vc=vc_, clf=final.named_steps["clf"], trained_on="train split (80%, seed 42)",
                     C=g_sv.best_params_["clf__C"]), out / "model_en.joblib")
    rep["model"] = {"checkpoint": "results/model_en.joblib", "features": int(final.named_steps["clf"].coef_.size),
                    "trainable_parameters": n_params(final), "fit_seconds_full_train": fit_s, "word_lr_fit_seconds": lr_fit}
    rep["timing"] = timing(final, [[x] for x in Xte], Xte, out / "_tmp_model.joblib"); os.remove(out / "_tmp_model.joblib")

    # --- near-duplicate leakage check
    mx, counts = near_duplicates(Xtr, Xte)
    keep = mx < 0.9
    rep["near_duplicates"] = {"method": "max cosine similarity (word 1-2-gram TF-IDF fit on train) of each test review to any train review",
                              "test_reviews_with_cosine": counts, "max_cosine_median": float(np.median(mx)),
                              "f1_macro_without_near_dups(>=0.9)": f1_score(yte[keep], pred[keep], average="macro"),
                              "n_kept": int(keep.sum())}
    # --- error examples (correct + incorrect)
    te2 = te.assign(pred=pred, score=scores(final, Xte))
    ex = pd.concat([te2[(te2.pred == te2.y) & (te2.y == 1)].sample(3, random_state=1), te2[(te2.pred == te2.y) & (te2.y == 0)].sample(3, random_state=1),
                    te2[(te2.pred != te2.y) & (te2.y == 1)].sample(3, random_state=1), te2[(te2.pred != te2.y) & (te2.y == 0)].sample(3, random_state=1)])
    ex.assign(outcome=np.where(ex.pred == ex.y, "correct", "wrong"), true=ex.y.map({0: "real", 1: "fake"}))[
        ["outcome", "true", "score", "category", "text"]].to_csv(out / "defense_examples_en.csv", index=False)
    json.dump(rep, open(out / "defense_report_en.json", "w"), indent=1, default=float)
    return rep


# ================================================================ RUSSIAN
def run_ru():
    import ru_data as R
    out = Path("results_ru"); df, ho = R.df.copy(), R.ho.copy()
    tr, te = train_test_split(df, test_size=0.2, stratify=df.generator, random_state=SEED)
    ytr, yte = tr.y.values, te.y.values
    rep = {"hardware": HW, "split": {"train": len(tr), "test": len(te), "seed": SEED, "stratified_by": "generator (rugpt3/qwen/real)",
                                     "test_by_generator": te.generator.value_counts().to_dict()}}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cols = lambda d: d[["text_ctrl", "lemmas"]]
    feats = ColumnTransformer([("lemma", word_vec(), "lemmas"), ("char", char_vec(), "text_ctrl")])
    proposed = Pipeline([("feats", feats), ("clf", LinearSVC(random_state=SEED))])
    lemma_lr = Pipeline([("feats", ColumnTransformer([("lemma", word_vec(), "lemmas")])), ("clf", LogisticRegression(max_iter=3000))])

    grid = [0.05, 0.1, 0.25, 0.5, 1, 2, 4]
    g_sv = GridSearchCV(proposed, {"clf__C": grid}, scoring="f1_macro", cv=cv, n_jobs=2, return_train_score=True).fit(cols(tr), ytr)
    g_lr = GridSearchCV(lemma_lr, {"clf__C": [0.5, 1, 2, 4, 8, 16]}, scoring="f1_macro", cv=cv, n_jobs=2).fit(cols(tr), ytr)
    rep["tuning"] = {"protocol": "5-fold stratified CV on train only, metric macro-F1",
                     "proposed_svm": {"grid": grid, "best_C": g_sv.best_params_["clf__C"], "cv_f1": g_sv.best_score_,
                                      "cv_f1_std": float(g_sv.cv_results_["std_test_score"][g_sv.best_index_])},
                     "lemma_lr": {"best_C": g_lr.best_params_["clf__C"], "cv_f1": g_lr.best_score_}}
    print("RU tuning", rep["tuning"], flush=True)
    lc = learning_curve(g_sv.best_estimator_, cols(tr), ytr, train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], cv=cv,
                        scoring="f1_macro", n_jobs=2, shuffle=True, random_state=SEED)
    plot_curves(lc, (grid, g_sv.cv_results_["mean_train_score"], g_sv.cv_results_["mean_test_score"]),
                "Russian: lemma + char TF-IDF + linear SVM (train split, 5-fold CV)", out / "defense_curves_ru.png")
    rep["learning_curve"] = {"train_sizes": lc[0].tolist(), "train_f1": lc[1].mean(1).tolist(), "val_f1": lc[2].mean(1).tolist()}

    a, b = train_test_split(np.arange(len(tr)), test_size=0.15, stratify=ytr, random_state=SEED)
    v = word_vec().fit(tr.lemmas.values[a])
    h = sgd_epoch_curve(v.transform(tr.lemmas.values[a]), ytr[a], v.transform(tr.lemmas.values[b]), ytr[b], 25,
                        out / "defense_epochs_ru.png", "Russian: SGD logistic regression, lemma TF-IDF (train 85% / validation 15%)")
    rep["sgd_epochs"] = h.round(4).to_dict("records")

    results = []
    results.append(full_metrics("Majority-class baseline", yte, np.full(len(yte), np.bincount(ytr).argmax()), np.zeros(len(yte))))
    S = lambda d: pd.DataFrame([R.stylo(x, True) for x in d.text_ctrl])
    sty = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]).fit(S(tr), ytr)
    results.append(full_metrics("Stylometric + LR (controlled)", yte, sty.predict(S(te)), sty.decision_function(S(te))))
    best_lr = g_lr.best_estimator_.fit(cols(tr), ytr)
    results.append(full_metrics(f"Lemma TF-IDF + LR (C={g_lr.best_params_['clf__C']})", yte, best_lr.predict(cols(te)), scores(best_lr, cols(te))))
    t = time.time(); final = g_sv.best_estimator_.fit(cols(tr), ytr); fit_s = time.time() - t
    pred = final.predict(cols(te))
    results.append(full_metrics(f"PROPOSED: lemma+char TF-IDF + linear SVM (C={g_sv.best_params_['clf__C']})", yte, pred, scores(final, cols(te))))
    res = pd.DataFrame(results); res.to_csv(out / "defense_results_ru.csv", index=False)
    print(res[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "test_n"]].round(4).to_string(), flush=True)
    plot_cm(results[-1]["confusion_matrix [[TN,FP],[FN,TP]]"], "Russian test set (proposed model)", out / "defense_cm_ru.png")
    rep["recall_by_generator_on_test"] = {g: float((pred[(te.generator == g).values] == (g != "real")).mean()) for g in ["rugpt3", "qwen", "real"]}

    # held-out modern-LLM probe (never used for training or selection)
    hp = final.predict(cols(ho))
    rep["modern_llm_probe"] = {"n_fake": int((ho.y == 1).sum()), "n_real": int((ho.y == 0).sum()),
                               "fake_recall": float((hp[ho.y.values == 1] == 1).mean()), "real_specificity": float((hp[ho.y.values == 0] == 0).mean()),
                               "f1_macro": f1_score(ho.y, hp, average="macro")}

    lem, ch = final.named_steps["feats"].named_transformers_["lemma"], final.named_steps["feats"].named_transformers_["char"]
    joblib.dump(dict(vl=lem, vc=ch, clf=final.named_steps["clf"], trained_on="train split (80%, seed 42)", C=g_sv.best_params_["clf__C"]),
                out / "model_ru.joblib")
    rep["model"] = {"checkpoint": "results_ru/model_ru.joblib", "features": int(final.named_steps["clf"].coef_.size),
                    "trainable_parameters": n_params(final), "fit_seconds_full_train": fit_s}
    rep["timing"] = timing(final, [te.iloc[[i]][["text_ctrl", "lemmas"]] for i in range(len(te))], cols(te), out / "_tmp.joblib"); os.remove(out / "_tmp.joblib")
    mx, counts = near_duplicates(tr.text_ctrl.values, te.text_ctrl.values)
    keep = mx < 0.9
    rep["near_duplicates"] = {"test_reviews_with_cosine": counts, "max_cosine_median": float(np.median(mx)),
                              "f1_macro_without_near_dups(>=0.9)": f1_score(yte[keep], pred[keep], average="macro"), "n_kept": int(keep.sum())}
    te2 = te.assign(pred=pred, score=scores(final, cols(te)))
    pick = lambda m, k: te2[m].sample(min(k, int(m.sum())), random_state=1)
    ex = pd.concat([pick((te2.pred == te2.y) & (te2.y == 1), 3), pick((te2.pred == te2.y) & (te2.y == 0), 3),
                    pick((te2.pred != te2.y) & (te2.y == 1), 3), pick((te2.pred != te2.y) & (te2.y == 0), 3)])
    ex.assign(outcome=np.where(ex.pred == ex.y, "correct", "wrong"), true=ex.y.map({0: "real", 1: "fake"}))[
        ["outcome", "true", "generator", "score", "review"]].to_csv(out / "defense_examples_ru.csv", index=False)
    json.dump(rep, open(out / "defense_report_ru.json", "w"), indent=1, default=float, ensure_ascii=False)
    return rep


def epochs_only():
    """Regenerate only the epoch-wise curves (they do not affect the reported test results)."""
    import experiments_core as E, ru_data as R
    tr, _ = train_test_split(E.df, test_size=0.2, stratify=E.df.y, random_state=SEED)
    X, y = tr.text_norm.to_numpy(dtype=object), tr.y.to_numpy()
    a, b = train_test_split(np.arange(len(X)), test_size=0.1, stratify=y, random_state=SEED)
    v = word_vec().fit(X[a])
    h = sgd_epoch_curve(v.transform(X[a]), y[a], v.transform(X[b]), y[b], 25, Path("results/defense_epochs_en.png"),
                        "English: SGD logistic regression, word TF-IDF (train 90% / validation 10% of train split)")
    r = json.load(open("results/defense_report_en.json")); r["sgd_epochs"] = h.round(4).to_dict("records"); json.dump(r, open("results/defense_report_en.json", "w"), indent=1)
    tr, _ = train_test_split(R.df, test_size=0.2, stratify=R.df.generator, random_state=SEED); y = tr.y.to_numpy()
    a, b = train_test_split(np.arange(len(tr)), test_size=0.15, stratify=y, random_state=SEED)
    v = word_vec().fit(tr.lemmas.values[a])
    h = sgd_epoch_curve(v.transform(tr.lemmas.values[a]), y[a], v.transform(tr.lemmas.values[b]), y[b], 25,
                        Path("results_ru/defense_epochs_ru.png"), "Russian: SGD logistic regression, lemma TF-IDF (train 85% / validation 15%)")
    r = json.load(open("results_ru/defense_report_ru.json")); r["sgd_epochs"] = h.round(4).to_dict("records")
    json.dump(r, open("results_ru/defense_report_ru.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["epochs"]: epochs_only(); sys.exit()
    which = sys.argv[1:] or ["ru", "en"]
    if "ru" in which: run_ru()
    if "en" in which: run_en()
    print("done")
