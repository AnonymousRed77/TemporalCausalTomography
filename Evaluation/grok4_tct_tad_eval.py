import os
import time
import csv
import re
import pandas as pd
import concurrent.futures
from xai_sdk import Client
from xai_sdk.chat import user, system

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

client = Client(
    timeout=3600,
)

MODEL_ID = "grok-4.20-non-reasoning"
MAX_WORKERS = 4  # Number of parallel evaluations

# TODO: Replace with actual path to your metadata.csv file here
metadata_file = 'path/to/metadata.csv'

# TODO: Replace with the path to your VLM result CSV file here
model_response_file = "/path/to/vlm/response/file.csv" 

# TODO: Replace 'model_name' with the VLM being evaluated here (e.g., llava_next, llava_cot, qwen3vl, etc.)
output_csv = 'model_name_tad_eval.csv'

# Initialize output CSV
if not os.path.exists(output_csv):
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

tad_tasks = []
with open(model_response_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['type'].lower() == 'tad':
            fname = os.path.basename(row['file_name']).strip('"')
            if fname in metadata:
                tad_tasks.append({
                    "file_name": fname,
                    "meta": metadata[fname],
                    "response": row['response']
                })

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
# WORKER FUNCTION
# ────────────────────────────────────────────────
def evaluate_one_item(task):
    input_text = template.format(
        process=task['meta']['temporal_casal_process'],
        band=task['meta']['temporal_scale_band'],
        question=task['meta']['tad_question'],
        flag=task['meta']['tad_answer'],
        model_response=task['response']
    )

    try:
        # Create chat instance
        chat = client.chat.create(model=MODEL_ID)
        chat.append(user(input_text))
        
        # Get the response object
        response = chat.sample()
        
        # EXTRACT THE STRING CONTENT
        if hasattr(response, 'outputs') and len(response.outputs) > 0:
            content = response.outputs[0].message.content
        else:
            # Fallback for different SDK versions
            content = str(response) 
            
        # Parse the brackets [A][B][C]
        matches = re.findall(r'\[([ABC])\]', content)
        
        if len(matches) == 3:
            return task['file_name'], matches[0], matches[1], matches[2]
        else:
            print(f"Format error for {task['file_name']}. Content received: {content}")
            return None
    except Exception as e:
        print(f"Error evaluating {task['file_name']}: {e}")
        return None

# ────────────────────────────────────────────────
# EXECUTION LOOP
# ────────────────────────────────────────────────
print(f"Starting Grok-4.20 reasoning evaluation for {len(tad_tasks)} items...")

processed_count = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Map tasks to the worker function
    future_to_file = {executor.submit(evaluate_one_item, task): task['file_name'] for task in tad_tasks}
    
    for future in concurrent.futures.as_completed(future_to_file):
        result = future.result()
        if result:
            fname, ctf, cg, rc = result
            with open(output_csv, 'a') as f:
                f.write(f"{fname},{ctf},{cg},{rc}\n")
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Progress: {processed_count}/{len(tad_tasks)} items evaluated.")

print(f"\nDone! Processed {processed_count} items into {output_csv}.")