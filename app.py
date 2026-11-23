"""Web demo for the bilingual fake-review detector.
Run:  py -m streamlit run app.py      (Mac: python3 -m streamlit run app.py)
Needs results_ru/model_ru.joblib (run experiments_ru.py once) and data/fake.csv (EN model trains on first use).
"""
import math
import joblib
import streamlit as st
from scipy.sparse import hstack
import predict as P

st.set_page_config(page_title="Fake Review Detector", page_icon="🔍", layout="centered")


@st.cache_resource(show_spinner="Loading Russian model…")
def ru_model():
    return joblib.load("results_ru/model_ru.joblib")


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

with st.expander("About the model & its limits"):
    st.markdown(
        "- ~0.96 macro-F1 on fakes from GPT-2 (English) and ruGPT-3 / Qwen2.5 (Russian).\n"
        "- Catches only ~17–31% of fakes written by a modern LLM: fluent, specific text often passes as real.\n"
        "- Can misjudge real reviews with typos or unusual phrasing.\n"
        "- Code & report: github.com/ai-naz/fake-review-detection"
    )
