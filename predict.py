import pandas as pd
import requests
import time
import re
import sys

# ========= SETTINGS =========
# Your input data file, assuming it's in your home directory (~)
CSV_PATH = "HellaSwagDataset.csv" 
# Output file path 
OUTPUT_PATH = #Enter path here 
# Models confirmed to be downloaded and ready for use
MODELS = #List of models 
NUM_ROWS = None
OLLAMA_API = "http://localhost:11434/api/generate" 

# ========= LOAD DATA =========
print(f"Loading data from {CSV_PATH}...")
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Error: {CSV_PATH} not found. Ensure the file is in the same directory as the script.")
    sys.exit(1)

# Add prediction columns for each model
for model in MODELS:
    df[model] = -1

# ========= LOOP =========
print(f"Starting prediction loop for {len(df)} rows across {len(MODELS)} models...")
for i, row in df.iterrows():
    # Context and the four ending options (using 1-indexed column names)
    ctx = row["ctx"]
    endings = [row["end1"], row["end2"], row["end3"], row["end4"]]
    
    # Prompt is constructed to use 1, 2, 3, 4 indices
    prompt = (
        "Given the following story context, choose which ending (1, 2, 3, or 4) is the most plausible and natural continuation.\\n"
        f"Context: {ctx}\\n"
        f"Ending 1: {endings[0]}\\n"
        f"Ending 2: {endings[1]}\\n"
        f"Ending 3: {endings[2]}\\n"
        f"Ending 4: {endings[3]}\\n"
        "Respond with ONLY the number (1, 2, 3, or 4). If the model chooses a number, we expect the output to be only that number, e.g., '1'."
    )

    print(f"\n--- Row {i+1}/{len(df)} ---")
    for model in MODELS:
        start_time = time.time()
        prediction = -1
        try:
            payload = {"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": 5}}
            
            response = requests.post(OLLAMA_API, json=payload, timeout=90)
            response.raise_for_status()
            
            text = response.json().get('response', '').strip()
            
            # Use regex to find the first digit (1, 2, 3, or 4)
            match = re.search(r"[1-4]", text)
            if match:
                prediction = int(match.group(0))
            else:
                prediction = -2 

            df.loc[i, model] = prediction

            print(f"  {model:<20} -> Prediction: {prediction} (Time: {time.time() - start_time:.2f}s)")
            
        except requests.exceptions.ConnectionError:
            print(f"  {model:<20} -> FAILED: Ollama server connection error.")
            df.loc[i, model] = -3
        except requests.exceptions.RequestException as e:
            print(f"  {model:<20} -> FAILED: API Request error ({e}).")
            df.loc[i, model] = -4

# ========= SAVE RESULTS =========
print(f"\nSaving results to {OUTPUT_PATH}...")
df.to_csv(OUTPUT_PATH, index=False)
print("Done.")
