import pandas as pd
import numpy as np
import re
import time
from collections import Counter

# ===================== CONFIG =====================

URL = "http://localhost:11434/api/generate"  # adjust if needed
N_RUNS = 5
PROGRESS_EVERY = 50

MODELS = [
#Model names
]

# =================================================

def load_model(model):
    # Optional: no-op if your server auto-loads
    print(f"[LOADING] {model}", flush=True)

def unload_model(model):
    # Optional: no-op or Ollama unload hook
    print(f"[UNLOADING] {model}", flush=True)

# ---------------- Load prompts ----------------
# df must contain a column named "prompt"
df = pd.read_csv("input_prompts.csv")
TOTAL_ROWS = len(df)

# predictions[model][row_idx] = [p1, p2, ..., p5]
predictions = {
    model: {idx: [] for idx in df.index}
    for model in MODELS
}

# ===================== INFERENCE =====================
for model in MODELS:
    print(f"\n================ MODEL: {model} ================", flush=True)
    load_model(model)

    for run in range(N_RUNS):
        print(f"\n[RUN {run+1}/{N_RUNS}] {model}", flush=True)
        start_time = time.time()

        for idx, row in df.iterrows():
            prompt = row["prompt"]

            try:
                r = requests.post(
                    URL,
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": 10,
                            # uncomment to test determinism
                            # "temperature": 0,
                        },
                    },
                    timeout=(10, 180),
                )
                r.raise_for_status()

                j = r.json()
                if "response" not in j:
                    raise ValueError(f"Malformed response: {j}")

                text = j["response"].strip()
                m = re.fullmatch(r"\s*([0-3])\s*", text)
                pred = int(m.group(1)) if m else -2

            except requests.exceptions.ConnectionError:
                pred = -3
            except Exception:
                pred = -4

            predictions[model][idx].append(pred)

            if (idx + 1) % PROGRESS_EVERY == 0:
                elapsed = time.time() - start_time
               rate = (idx + 1) / elapsed
                print(
                    f"[{model} | run {run+1}] "
                    f"{idx+1}/{TOTAL_ROWS} "
                    f"({100*(idx+1)/TOTAL_ROWS:.1f}%) "
                    f"| {rate:.2f} rows/sec",
                    flush=True
                )

    unload_model(model)
    print(f"[MODEL DONE] {model}", flush=True)

# ===================== ANALYSIS =====================

def analyze_model(pred_dict):
    agreements = []
    full_agree = 0
    any_error = 0

    for preds in pred_dict.values():
        if any(p < 0 for p in preds):
            any_error += 1

        counts = Counter(preds)
        mode_pred, mode_freq = counts.most_common(1)[0]
        agreement = mode_freq / len(preds)

        agreements.append(agreement)
        if agreement == 1.0:
            full_agree += 1

    return {
        "rows": len(pred_dict),
        "full_agreement_pct": 100 * full_agree / len(pred_dict),
        "mean_agreement": float(np.mean(agreements)),
        "any_error_pct": 100 * any_error / len(pred_dict),
    }

# ===================== SUMMARY =====================

print("\n================ STABILITY SUMMARY ================")
print(
    f"{'MODEL':18s} | "
    f"{'FULL_AGREE %':>12s} | "
    f"{'MEAN_AGREE':>10s} | "
    f"{'ERROR %':>8s}"
)

print("-" * 60)

for model in MODELS:
    stats = analyze_model(predictions[model])
    print(
        f"{model:18s} | "
        f"{stats['full_agreement_pct']:12.2f} | "
        f"{stats['mean_agreement']:10.3f} | "
        f"{stats['any_error_pct']:8.2f}"
    )

print("\n[ALL MODELS COMPLETED]")
