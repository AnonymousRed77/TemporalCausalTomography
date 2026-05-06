from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import pandas as pd

MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"

file_name = "qwen3vl_30b_tct.csv"

# TODO: Replace with actual path to your metadata.csv file here
metadata_path = 'path/to/metadata.csv'

# TODO: Replace with actual path to your TCT images directory here
image_base_dir = "/path/to/TCT-dataset/images/"

# Load model and processor
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Prepare output CSV
with open(file_name, 'w', encoding='utf-8') as f:
    f.write("file_name,type,prompt,response\n")

df = pd.read_csv(metadata_path)

count = 1
for index, row in df.iterrows():
    print(f"\n=== Processing Image {count}: {row['file_name']} ===")
    
    image_path = image_base_dir + row['file_name']
    image = Image.open(image_path).convert('RGB')

    queries = [
        row['tsc_question'],
        row['tad_question'],
        row['btp_question'],
        row['tcap_question']
    ]
    types_list = ["tsc", "tad", "btp", "tcap"]

    # Start fresh conversation for this image
    conversation = []

    for i in range(len(queries)):

        question = queries[i] + " Output your multiple-choice answer first (A, B, C, D, E, or F), then explain your reasoning in 2-3 sentences."

        if i == 0:
            # First turn: include the image
            user_message = {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        else:
            # Subsequent turns: text only
            user_message = {
                "role": "user",
                "content": [{"type": "text", "text": question}]
            }

        conversation.append(user_message)

        # Prepare inputs
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Generate
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,      
            do_sample=False,
            temperature=0.0
        )

        # Trim input tokens and decode only the new response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        response = output_text[0].strip()

        # Save to CSV (safe escaping)
        with open(file_name, 'a', encoding='utf-8') as f:
            safe_prompt = queries[i].replace('"', '""')
            safe_response = response.replace('"', '""')
            f.write(f'{image_path},"{types_list[i]}","{safe_prompt}","{safe_response}"\n')

        print(f"[{types_list[i].upper()}] Response:\n{response}\n")

        # Add assistant response to conversation history
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })

    count += 1

print(f"\nFinished! Results saved to {file_name}")