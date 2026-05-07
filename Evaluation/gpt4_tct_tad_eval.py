import os
import time
import json
import csv
import re
from openai import OpenAI

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
client = OpenAI() 

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

# Load Metadata into a dictionary for quick lookup
metadata = {}
with open(metadata_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        metadata[row['file_name']] = row

# Load Model Responses (TAD Only)
tad_tasks = []
with open(model_response_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['type'].lower() == 'tad':
            # Clean filename to match metadata
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
# 1. Build batch requests
# ────────────────────────────────────────────────
batch_requests = []

for idx, task in enumerate(tad_tasks):
    # Extract model's choice letter (usually the first character of response)
    model_full_response = task['response']
    
    input_text = template.format(
        process=task['meta']['temporal_casal_process'],
        band=task['meta']['temporal_scale_band'],
        question=task['meta']['tad_question'],
        flag=task['meta']['tad_answer'],
        model_response=model_full_response
    )

    request_obj = {
        "custom_id": f"tad-{idx}-{task['file_name']}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": input_text}],
            "max_tokens": 10,
            "temperature": 0.0
        }
    }
    batch_requests.append(request_obj)

# Write to JSONL
input_jsonl = "batch_input_tad_logic.jsonl"
with open(input_jsonl, "w") as f:
    for req in batch_requests:
        f.write(json.dumps(req) + "\n")

print(f"Created batch input file with {len(batch_requests)} requests.")

# ────────────────────────────────────────────────
# 2. Upload and Create Batch
# ────────────────────────────────────────────────
batch_file = client.files.create(file=open(input_jsonl, "rb"), purpose="batch")
batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
print(f"Batch job created: {batch_job.id}")

# ────────────────────────────────────────────────
# 3. Poll for completion
# ────────────────────────────────────────────────
while True:
    batch_job = client.batches.retrieve(batch_job.id)
    print(f"Status: {batch_job.status}...")
    if batch_job.status in ["completed", "failed", "expired", "cancelled"]:
        break
    time.sleep(30)

if batch_job.status != "completed":
    raise RuntimeError(f"Batch failed: {batch_job.status}")

# ────────────────────────────────────────────────
# 4. Process Output
# ────────────────────────────────────────────────
output_file = client.files.content(batch_job.output_file_id)
output_lines = output_file.text.splitlines()

processed_count = 0
for line in output_lines:
    result = json.loads(line)
    custom_id = result.get("custom_id")
    file_name = custom_id.split("-", 2)[-1] # Extract file_name from custom_id

    try:
        content = result["response"]["body"]["choices"][0]["message"]["content"]
        # Extract letters inside brackets [A][C][B]
        matches = re.findall(r'\[([ABC])\]', content)
        
        if len(matches) == 3:
            ctf, cg, rc = matches[0], matches[1], matches[2]
            with open(output_csv, 'a') as f:
                f.write(f"{file_name},{ctf},{cg},{rc}\n")
            processed_count += 1
    except Exception as e:
        print(f"Error parsing {custom_id}: {e}")

print(f"Done! Processed {processed_count} items into {output_csv}.")