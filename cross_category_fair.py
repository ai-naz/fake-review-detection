"""Size-matched cross-category test: train on N reviews from OTHER categories vs N from the SAME category."""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import experiments_core as ec
df = ec.df; rows = []
for cat in sorted(df.category.unique()):
    b = df[df.category == cat]
    a2, b2 = train_test_split(b, test_size=0.2, stratify=b.y, random_state=42)   # same test set for both
    o = df[df.category != cat]; other = pd.concat([o[o.y == k].sample(len(a2)//2, random_state=42) for k in (0, 1)])
    m_in, *_ = ec.evaluate("tfidf_word_lr", a2, b2, controlled=True)
    m_x, *_ = ec.evaluate("tfidf_word_lr", other, b2, controlled=True)
    m_all, *_ = ec.evaluate("tfidf_word_lr", df[df.category != cat], b2, controlled=True)
    rows.append(dict(held_out=cat, in_domain_f1=m_in["macro_f1"], cross_f1_size_matched=m_x["macro_f1"], cross_f1_all_data=m_all["macro_f1"], n_train_matched=len(a2)))
    print(rows[-1])
r = pd.DataFrame(rows); r["gap"] = r.cross_f1_size_matched - r.in_domain_f1
r.to_csv("results/cross_category.csv", index=False); print(r[["in_domain_f1","cross_f1_size_matched","cross_f1_all_data","gap"]].mean())
l = r.sort_values("gap"); y = np.arange(len(l))
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(y - 0.27, l.in_domain_f1, 0.27, label="same category", color="#ADB5BD")
ax.barh(y, l.cross_f1_size_matched, 0.27, label="other categories (same size)", color="#4C6EF5")
ax.barh(y + 0.27, l.cross_f1_all_data, 0.27, label="other categories (all ~36k)", color="#F08C00")
ax.set_yticks(y, l.held_out.str.replace("_", " ")); ax.set_xlim(0.7, 1); ax.set_xlabel("Macro-F1 on held-out category test set")
ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3); fig.tight_layout(); fig.savefig("results/fig_cross_category.png", dpi=200)
