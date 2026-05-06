import os
import torch
from PIL import Image
import pandas as pd
import re
from transformers import MllamaForConditionalGeneration, AutoProcessor

# ========================= CONFIG =========================
MODEL_NAME = "Xkev/Llama-3.2V-11B-cot"

file_name = "llava_cot_tct.csv"

# TODO: Replace with actual path to your metadata.csv file here
metadata_path = 'path/to/metadata.csv'

# TODO: Replace with actual path to your TCT images directory here
image_base_dir = "/path/to/TCT-dataset/images/"

MULTI_TURN = True

# ========================= LOAD MODEL =========================
model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

# ========================= OUTPUT FILE =========================
with open(file_name, 'w', encoding='utf-8') as f:
    f.write("file_name,type,prompt,response\n")

df = pd.read_csv(metadata_path)

# ========================= MAIN LOOP =========================
for index, row in df.iterrows():
    print(f"\n=== Processing {row['file_name']} ===")

    image_path = os.path.join(image_base_dir, row['file_name'])

    # Safe image loading
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        continue

    queries = [
        row['tsc_question'],
        row['tad_question'],
        row['btp_question'],
        row['tcap_question']
    ]
    types_list = ["tsc", "tad", "btp", "tcap"]

    # Conversation history
    conversation = []

    for i in range(len(queries)):

        question = queries[i] + " Output your multiple-choice answer first (A, B, C, D, E, or F), then explain your reasoning in 2-3 sentences."

        # Reset conversation if single-turn
        if not MULTI_TURN:
            conversation = []

        # First turn includes image
        if i == 0 or not MULTI_TURN:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        else:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": question}
                ]
            }

        conversation.append(user_message)

        # Build prompt
        input_text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )


        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        # ========================= RESPONSE EXTRACTION =========================
        decoded = processor.decode(output[0], skip_special_tokens=True)

        # Prefer <CONCLUSION> tag if exists
        match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", decoded, re.DOTALL)

        if match:
            response = match.group(1).strip()
        else:
            # fallback: remove prompt
            generated_ids = output[0][inputs.input_ids.shape[-1]:]
            response = processor.decode(
                generated_ids,
                skip_special_tokens=True
            ).strip()

        print(f"[{types_list[i].upper()}] {response}\n")

        # ========================= CSV WRITE =========================
        safe_prompt = queries[i].replace('"', '""')
        safe_response = response.replace('"', '""')

        with open(file_name, 'a', encoding='utf-8') as f:
            f.write(f'{image_path},"{types_list[i]}","{safe_prompt}","{safe_response}"\n')

        # Add assistant response (for multi-turn)
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })

print(f"\nFinished! Saved to {file_name}")