# Fake Review Detection (English & Russian, classical NLP)

Binary text classification: is a product review **real** (human-written) or **fake** (machine-generated)?
Classical NLP only: TF-IDF (word/lemma + character n-grams) → linear SVM. No LLM in the detector.

**Held-out test results (macro-F1):** English **0.956** (8,083 reviews; baseline 0.859) · Russian **0.956** (409 reviews; baseline 0.726).
**Main limitation:** on reviews written by a modern LLM, the Russian model catches only **28%** of fakes.

Full details, dataset cards, configs and the result tables are in **[TECHNICAL.md](TECHNICAL.md)**; the write-up is in `Fake_Review_Detection_Report_EN_RU.pdf`.

## Setup
```bash
pip install -r requirements.txt          # Python 3.13/3.14, CPU only
```
Windows: use `py` instead of `python` and `curl.exe` instead of `curl`.

## Data (not redistributed: download once)
```bash
curl -L -o data/fake.csv "https://raw.githubusercontent.com/SayamAlt/Fake-Reviews-Detection/main/fake%20reviews%20dataset.csv"
curl -L -o data_ru/rureviews.csv "https://raw.githubusercontent.com/sismetanin/rureviews/master/women-clothing-accessories.3-class.balanced.csv"
```
- English: Salminen et al. (2022), 40,412 Amazon reviews after cleaning, 50% real / 50% GPT-2-generated. Original: https://osf.io/tyue9/
- Russian: 1,022 real reviews (RuReviews) + 1,022 fakes we generated with ruGPT-3 and Qwen2.5 (`data_ru/fake_*.csv`, included). Plus a 162 + 162 modern-LLM probe set (`data_ru/fake_claude.txt`).

## Training and evaluation (reported numbers)
```bash
python defense_eval.py ru    # ~1 min
python defense_eval.py en    # ~15-25 min on a laptop CPU
```
Protocol: fixed stratified 80/20 train/test split (seed 42) → hyperparameter C chosen by cross-validation **on train only** → refit on train → **one** evaluation on test.
Outputs: `results*/defense_results_*.csv` (metrics table), `defense_report_*.json` (config, tuning, timing, near-duplicate check), `defense_curves_*.png` (learning and validation curves), `defense_epochs_*.png` (loss per epoch), `defense_cm_*.png` (confusion matrices), `defense_examples_*.csv` (correct and incorrect examples), and the checkpoints `results/model_en.joblib`, `results_ru/model_ru.joblib`.

## Inference
```bash
python predict.py "Отличное платье, рекомендую!" "Great product, works as described."
python -m streamlit run app.py            # web demo at http://localhost:8501 (shows the preprocessing and top features)
```
Score > 0 → fake, < 0 → real; values near 0 mean the model is unsure.

## Exploratory analyses
| Script | What it shows |
|---|---|
| `experiments.py` (~4 min) | English model comparison, punctuation-shortcut ablation, top n-grams, errors by length |
| `cross_category_fair.py` | Size-matched test on unseen product categories |
| `experiments_ru.py` | Russian 5-fold CV, leave-one-generator-out, English model applied to Russian |
| `make_report.py` | Builds the PDF report |
| `generate_ru.py` | Regenerates the Russian fakes (`pip install -r requirements-generate.txt`, ~35 min CPU) |
