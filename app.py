"""Web demo for the bilingual fake-review detector.
Run:  py -m streamlit run app.py      (Mac: python3 -m streamlit run app.py)
Needs the checkpoints from defense_eval.py (results/model_en.joblib, results_ru/model_ru.joblib).
"""
import math
import streamlit as st
from scipy.sparse import hstack
import predict as P

st.set_page_config(page_title="Fake Review Detector", page_icon="🔍", layout="centered")


@st.cache_resource(show_spinner="Loading Russian model…")
def ru_model():
    return P.load_ru()


@st.cache_resource(show_spinner="Loading English model (first run trains it, ~1–2 min)…")
def en_model():
    return P.load_en()


def classify(text):
    if P.is_russian(text):
        m, c = ru_model(), P.control_ru(text)
        X = hstack([m["vl"].transform([P.lemmatize(c)]), m["vc"].transform([c])])
        lang = "Russian"
    else:
        m, c = en_model(), P.control_en(text)
        X = hstack([m["vw"].transform([c]), m["vc"].transform([c])])
        lang = "English"
    return lang, float(m["clf"].decision_function(X)[0])


st.title("🔍 Fake Review Detector")
st.caption("Classical NLP (TF-IDF + linear SVM) · English & Russian · NLP2, KBTU")

examples = {
    "— pick an example —": "",
    "RU · real-style": "качество норм но рукава длинные пришлось подшивать",
    "RU · generic (modern-LLM style)": "Отличное платье, ткань приятная, размер подошел идеально. Рекомендую!",
    "EN · short generic": "Great product, works as described.",
    "EN · detailed": "I was impressed with the quality for the price but it was just way too small, had to return it.",
}
choice = st.selectbox("Examples", list(examples))
text = st.text_area("Paste a product review (English or Russian):", value=examples[choice], height=140)

if st.button("Check review", type="primary"):
    if not text.strip():
        st.info("Type or paste a review first.")
    else:
        lang, s = classify(text.strip())
        p_fake = 1 / (1 + math.exp(-2 * s))  # squashed SVM score for display only, not a calibrated probability
        if s > 0:
            st.error(f"**Likely FAKE (machine-generated)** · {lang} model")
        else:
            st.success(f"**Likely REAL** · {lang} model")
        st.progress(p_fake, text=f"Fake-ness: {p_fake:.0%}")
        st.caption(f"Raw SVM score: {s:+.3f} (above 0 = fake, below 0 = real; near 0 = unsure)")
        if abs(s) < 0.15:
            st.warning("The score is close to 0, so the model is unsure. Short or generic reviews are hard.")
        with st.expander("Show the pipeline: preprocessing → tokens → features → score", expanded=True):
            ex = P.explain(text.strip())
            steps = [f"**Language detected:** {ex['language']}", f"**Normalized text:** `{ex['normalized']}`"]
            if "lemmas" in ex:
                steps.append(f"**Lemmas (pymorphy3):** `{ex['lemmas']}`")
            steps += [f"**Word tokens / n-grams (first 20):** `{', '.join(ex['word tokens'])}`",
                      f"**TF-IDF vector:** {ex['non-zero features']} non-zero features (word/lemma n-grams + character 2–5-grams)"]
            for i, line in enumerate(steps, 1):
                st.markdown(f"{i}. {line}")
            c1, c2 = st.columns(2)
            c1.markdown("**Pushes toward FAKE**"); c1.table({"feature": [f for f, _ in ex["toward FAKE"]], "weight": [w for _, w in ex["toward FAKE"]]})
            c2.markdown("**Pushes toward REAL**"); c2.table({"feature": [f for f, _ in ex["toward REAL"]], "weight": [w for _, w in ex["toward REAL"]]})
            st.caption("Score = sum of (TF-IDF value × learned weight) over all features + bias. Above 0 → fake.")

with st.expander("About the model & its limits"):
    st.markdown(
        "- Held-out test macro-F1: 0.956 English (8,083 reviews, GPT-2 fakes), 0.956 Russian (409 reviews, ruGPT-3/Qwen fakes).\n"
        "- Catches only 28% of fakes written by a modern LLM (162-review probe): fluent, specific text often passes as real.\n"
        "- Can misjudge real reviews with typos or unusual phrasing.\n"
        "- Code & report: github.com/ai-naz/fake-review-detection"
    )
