# Fake review detection (classical NLP)
Data: Salminen et al. (2022) fake reviews dataset. Download it to data/fake.csv:
https://raw.githubusercontent.com/SayamAlt/Fake-Reviews-Detection/main/fake%20reviews%20dataset.csv
(original: https://osf.io/tyue9/)

pip install pandas scikit-learn matplotlib reportlab
python experiments.py          # ~4 min CPU: models, ablation, n-grams, errors
python cross_category_fair.py  # size-matched unseen-category test
python make_report.py          # builds Fake_Review_Detection_Report.pdf
