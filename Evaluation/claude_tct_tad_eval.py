import os
import time
import json
import csv
import re
import pandas as pd
import anthropic
import logging

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

client = anthropic.Anthropic()

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
# DATA LOADING (Merging Metadata + Model Responses)
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
                    "response": row['response'],
                    "prompt": row['prompt']
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
# 1. Build Batch Requests
# ────────────────────────────────────────────────
requests = []
custom_id_to_file = {}

for idx, task in enumerate(tad_tasks):
    input_text = template.format(
        process=task['meta']['temporal_casal_process'],
        band=task['meta']['temporal_scale_band'],
        question=task['meta']['tad_question'],
        flag=task['meta']['tad_answer'],
        model_response=task['response']
    )

    sanitized_filename = task['file_name'].replace('.', '_')
    custom_id = f"tad_{idx}_{sanitized_filename}"
    
    # Ensure ID doesn't exceed 64 characters (Anthropic limit)
    if len(custom_id) > 64:
        custom_id = custom_id[:64]

    # Anthropic Batch API Structure
    request = {
        "custom_id": custom_id,
        "params": {
            "model": "claude-sonnet-4-6",
            "max_tokens": 10,
            "temperature": 0.0,
            "messages": [
                {"role": "user", "content": input_text}
            ]
        }
    }
    requests.append(request)
    custom_id_to_file[custom_id] = task['file_name'] # Maps back to original filename with the dot

print(f"Prepared {len(requests)} requests for Claude Batch.")

# ────────────────────────────────────────────────
# 2. Create Batch Job
# ────────────────────────────────────────────────
batch = client.messages.batches.create(requests=requests)
print(f"Batch created with ID: {batch.id}")

# ────────────────────────────────────────────────
# 3. Poll for Completion
# ────────────────────────────────────────────────
while True:
    batch_status = client.messages.batches.retrieve(batch.id)
    print(f"Status: {batch_status.processing_status} | "
          f"Succeeded: {batch_status.request_counts.succeeded} | "
          f"Failed: {batch_status.request_counts.errored}")

    if batch_status.processing_status in ["ended", "failed", "canceled"]:
        break
    time.sleep(30)

if batch_status.processing_status != "ended":
    raise ValueError(f"Batch failed with status: {batch_status.processing_status}")

# ────────────────────────────────────────────────
# 4. Retrieve Results and Save to CSV
# ────────────────────────────────────────────────
processed_count = 0

for result in client.messages.batches.results(batch.id):
    custom_id = result.custom_id
    file_name = custom_id_to_file.get(custom_id)
    
    if result.result.type != "succeeded":
        print(f"Request failed for {custom_id}")
        continue

    try:
        content = result.result.message.content[0].text
        # Extract letters inside brackets [A][C][B]
        matches = re.findall(r'\[([ABC])\]', content)
        
        if len(matches) == 3:
            ctf, cg, rc = matches[0], matches[1], matches[2]
            with open(output_csv, 'a') as f:
                f.write(f"{file_name},{ctf},{cg},{rc}\n")
            processed_count += 1
        else:
            print(f"Formatting error in response for {file_name}: {content}")

    except Exception as e:
        print(f"Failed to parse {custom_id}: {e}")

print(f"\nAll results processed. Saved {processed_count} evaluations to {output_csv}.")