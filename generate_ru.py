"""Generate Russian machine-written reviews with two open models, conditioned on real reviews' sentiment + length.
G1 = ai-forever/rugpt3small_based_on_gpt2  (GPT-2-era continuation model — mirrors the English dataset)
G2 = Qwen/Qwen2.5-0.5B-Instruct            (modern instruction-tuned chat model)
Usage: python generate_ru.py <gen> <n> [batch]
"""
import sys, re, time, random, html
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(2); random.seed(0); torch.manual_seed(0)
gen, N = sys.argv[1], int(sys.argv[2]); BS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
real = pd.read_csv("data_ru/rureviews.csv", sep="\t").dropna()
real["n"] = real.review.str.split().str.len()
real = real[(real.n >= 3) & (real.n <= 80)]
ref = real.sample(N, random_state={"rugpt3": 1, "qwen": 2}[gen])   # reference rows for sentiment + length only

TONE = {"positive": ("положительный", 5), "neautral": ("нейтральный, со смешанными впечатлениями", 3), "negative": ("отрицательный", 1)}
ITEMS = ["платье", "футболку", "куртку", "сумку", "джинсы", "кофту", "юбку", "шарф", "кроссовки", "рюкзак", "блузку", "серьги", "часы", "пальто", "купальник"]

if gen == "rugpt3":
    name = "ai-forever/rugpt3small_based_on_gpt2"
    def prompt(r):
        t, stars = TONE[r.sentiment]
        return f"Отзыв покупателя интернет-магазина. Товар: {random.choice(ITEMS)}. Оценка: {stars} из 5.\nТекст отзыва:"
else:
    name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
if gen == "qwen":
    def prompt(r):
        t, _ = TONE[r.sentiment]
        msg = (f"Напиши короткий отзыв покупателя интернет-магазина о покупке: {random.choice(ITEMS)}. "
               f"Тон: {t}. Длина: примерно {r.n} слов. Пиши как обычный покупатель, без заголовков и кавычек. Выведи только текст отзыва.")
        return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)

def clean(t, n_target):
    t = html.unescape(re.sub(r"\s+", " ", t)).strip().strip('"«»').strip()
    t = re.split(r"(?:Отзыв покупателя|Текст отзыва:|Оценка:|\n)", t)[0].strip()
    w = t.split()
    if len(w) > n_target * 1.6 + 5:                 # trim overly long outputs to a sentence boundary near target length
        t = " ".join(w[: int(n_target * 1.6) + 5])
    m = re.match(r"^(.*[.!?)])", t, re.S)           # cut unfinished trailing sentence (avoid the English truncation artifact)
    if m and len(m.group(1).split()) >= 3: t = m.group(1)
    return t

out, t0 = [], time.time()
rows = list(ref.itertuples())
for i in range(0, len(rows), BS):
    batch = rows[i:i + BS]
    enc = tok([prompt(r) for r in batch], return_tensors="pt", padding=True)
    maxnew = int(min(160, max(r.n for r in batch) * 3 + 20))
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=maxnew, do_sample=True, temperature=0.9, top_p=0.95,
                           repetition_penalty=1.15, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(o[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    for r, t in zip(batch, texts):
        out.append(dict(review=clean(t, r.n), sentiment=r.sentiment, generator=gen))
    print(f"{gen}: {len(out)}/{N}  {time.time()-t0:.0f}s", flush=True)
pd.DataFrame(out).to_csv(f"data_ru/fake_{gen}.csv", index=False)
