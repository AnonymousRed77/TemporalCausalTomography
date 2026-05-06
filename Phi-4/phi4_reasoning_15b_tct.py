import os
import torch
import pandas as pd
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# ========================= CONFIG =========================
MODEL_NAME = "microsoft/Phi-4-reasoning-vision-15B"

# TODO: Replace with actual path to your metadata.csv file here
metadata_path = 'path/to/metadata.csv'

# TODO: Replace with actual path to your TCT images directory here
image_base_dir = "/path/to/TCT-dataset/images/"

output_file = "phi4_reasoning_15b_tct.csv"

# ========================= LOAD MODEL =========================
print("Loading Phi-4 model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True,
).eval()

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

tokenizer = processor.tokenizer

print("Model loaded.")

# ========================= OUTPUT FILE =========================
with open(output_file, "w", encoding="utf-8") as f:
    f.write("file_name,type,prompt,response\n")

# ========================= LOAD DATA =========================
df = pd.read_csv(metadata_path)

# ========================= MAIN LOOP =========================
for idx, row in df.iterrows():

    print(f"\n=== Processing {row['file_name']} ===")

    image_path = os.path.join(image_base_dir, row["file_name"])

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to load {image_path}: {e}")
        continue

    questions = [
        row["tsc_question"],
        row["tad_question"],
        row["btp_question"],
        row["tcap_question"]
    ]

    types = ["tsc", "tad", "btp", "tcap"]

    # ========================= PER QUESTION =========================
    for i in range(len(questions)):

        question = questions[i] + " Output your multiple-choice answer first (A, B, C, D, E, or F), then explain your reasoning in 2-3 sentences."

        messages = [
            {
                "role": "user",
                "content": f"<image>\n{question}"
            }
        ]

        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        ).to("cuda")

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )

        # ========================= DECODE =========================
        gen_tokens = output[:, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

        print(f"[{types[i].upper()}] {response}\n")

        # ========================= SAVE =========================
        safe_prompt = questions[i].replace('"', '""')
        safe_response = response.replace('"', '""')

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f'{image_path},"{types[i]}","{safe_prompt}","{safe_response}"\n')

print(f"\nDone! Saved to {output_file}")