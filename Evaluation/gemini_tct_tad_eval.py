import os
import time
import json
import csv
import re
import pandas as pd
import threading
import concurrent.futures
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
client = genai.Client()
MODEL_ID = "gemini-2.5-flash"

# TODO: Replace with actual path to your metadata.csv file here
metadata_file = 'path/to/metadata.csv'

# TODO: Replace with the path to your VLM result CSV file here
model_response_file = "/path/to/vlm/response/file.csv" 

# TODO: Replace 'model_name' with the VLM being evaluated here (e.g., llava_next, llava_cot, qwen3vl, etc.)
output_csv = 'model_name_tad_eval.csv'

# Initialize output CSV
with open(output_csv, 'w') as f:
    f.write("file_name,ctf,cg,rc\n")

# ────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────
metadata = {}
with open(metadata_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        metadata[row['file_name']] = row

tasks = []
with open(model_response_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['type'].lower() == 'tad':
            fname = os.path.basename(row['file_name']).strip('"')
            if fname in metadata:
                tasks.append({
                    "file_name": fname,
                    "meta": metadata[fname],
                    "response": row['response'],
                    "prompt": row['prompt']
                })

# ────────────────────────────────────────────────
# RATE LIMIT SETTINGS
# ────────────────────────────────────────────────
MAX_TOKENS_PER_MINUTE = 800_000
MAX_WORKERS = 5
lock = threading.Lock()
tokens_used_this_minute = 0
last_reset_time = time.time()

# ────────────────────────────────────────────────
# PROMPT TEMPLATE
# ────────────────────────────────────────────────
template = '''You are a logic judge evaluating a Vision-Language Model's (VLM) explanation of a temporal scene. You will be provided with the context of an image and the VLM's explanation to a question about the image.

### Image Context:
- Causal Process: {process}
- Temporal Range: {band}
- Question: {question}
- Evaluation Flag: {flag}

### Candidate VLM's Response to Question:
{model_response}

### Task:
Evaluate the response on three dimensions. Output your answer in exactly this format: [CTF][CG][RC]
Each bracket must contain A (for Yes), B (for No), or C (for Not Applicable).

### CRITICAL RULES:
1. ZERO EXPLANATION: If the Candidate VLM's Response contains ONLY a choice (e.g., "A", "B", "A. Yes", "B. No") and provides NO supporting textual explanation or reasoning, you MUST output [C][C][C] and stop. Do not evaluate the dimensions individually if the explanation is missing.
2. CG DIMENSION CONDITIONAL: 
   - If Evaluation Flag is 'A', the second bracket MUST be [C]. 
   - If Evaluation Flag is 'B', evaluate the CG dimension as [A] or [B].
3. STRICT FORMAT: Do not explain your reasoning. Do not provide any conversational text. Just provide the three brackets.

Dimensions:
1. CTF (Causal-Temporal Faithfulness): Does the textual explanation reference a specific causal process (such as oxidation, evaporation, biological growth, erosion, thermal dynamics) and place its duration within the correct order-of-magnitude temporal range?
2. CG (Constraint Grounding): Does the textual explanation cite a specific physical, biological, or chemical constraint that makes the claimed timeline impossible? (Note: Refer to Critical Rule #2)
3. RC (Reasoning-Choice Alignment): Does the textual explanation logically support the model's selected A (Yes) or B (No) choice?

Example Output for Flag 'A': [A][C][A]
Example Output for Flag 'B': [A][B][A]

Output the bracketed letters and nothing else.'''

# ────────────────────────────────────────────────
# PROCESSING FUNCTION
# ────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def process_one(task):
    global tokens_used_this_minute, last_reset_time

    # Construct prompt
    input_text = template.format(
        process=task['meta']['temporal_casal_process'],
        band=task['meta']['temporal_scale_band'],
        question=task['meta']['tad_question'],
        flag=task['meta']['tad_answer'],
        model_response=task['response']
    )

    # Estimate tokens (Prompt + margin)
    est_tokens = len(input_text.split()) * 1.5 + 200

    # Rate limiting
    with lock:
        now = time.time()
        if now - last_reset_time >= 60:
            tokens_used_this_minute = 0
            last_reset_time = now
        
        if tokens_used_this_minute + est_tokens > MAX_TOKENS_PER_MINUTE:
            sleep_time = max(0, 60 - (now - last_reset_time))
            time.sleep(sleep_time + 1)
            tokens_used_this_minute = 0
            last_reset_time = time.time()
        
        tokens_used_this_minute += est_tokens

    # Generate
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=input_text,
        config=types.GenerateContentConfig(
            max_output_tokens=10,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )

    text = response.text.strip()
    
    # Parse Result
    matches = re.findall(r'\[([ABC])\]', text)
    if len(matches) == 3:
        return task['file_name'], matches
    else:
        # Fallback if model adds fluff
        return task['file_name'], None

# ────────────────────────────────────────────────
# MAIN EXECUTION
# ────────────────────────────────────────────────
print(f"Starting evaluation of {len(tasks)} items...")

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_file = {executor.submit(process_one, t): t['file_name'] for t in tasks}

    processed_count = 0
    for future in concurrent.futures.as_completed(future_to_file):
        file_name = future_to_file[future]
        try:
            fname, ratings = future.result()
            if ratings:
                ctf, cg, rc = ratings
                with open(output_csv, 'a') as f:
                    f.write(f"{fname},{ctf},{cg},{rc}\n")
                processed_count += 1
            else:
                print(f"Warning: Formatting error for {file_name}")
        except Exception as exc:
            print(f"Error processing {file_name}: {exc}")

print(f"\nDone! Processed {processed_count} items into {output_csv}.")