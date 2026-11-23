# Fake review detection (classical NLP)
Data: Salminen et al. (2022) fake reviews dataset. Download it to data/fake.csv:
https://raw.githubusercontent.com/SayamAlt/Fake-Reviews-Detection/main/fake%20reviews%20dataset.csv
(original: https://osf.io/tyue9/)

pip install pandas scikit-learn matplotlib reportlab
python experiments.py          # ~4 min CPU: models, ablation, n-grams, errors
python cross_category_fair.py  # size-matched unseen-category test
python make_report.py          # builds Fake_Review_Detection_Report.pdf

## Russian extension
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers pymorphy3 joblib
Real reviews -> data_ru/rureviews.csv:
https://raw.githubusercontent.com/sismetanin/rureviews/master/women-clothing-accessories.3-class.balanced.csv
python generate_ru.py rugpt3 800 16 ; python generate_ru.py qwen 288 16   # ~35 min CPU (generated files included in zip)
python experiments_ru.py                                                  # results_ru/ + model_ru.joblib

## Bilingual prediction (EN + RU)
python predict.py "Отличное платье, очень довольна!" "Great product, works as described."

## Web demo (Streamlit)
pip install streamlit
python -m streamlit run app.py        # Windows: py -m streamlit run app.py
# opens http://localhost:8501 — needs results_ru/model_ru.joblib (run experiments_ru.py first)
