from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import pandas as pd

MODEL_NAME = "llava-hf/llava-v1.6-vicuna-13b-hf"

file_name = "llavanext_13b_tct.csv"

# TODO: Replace with actual path to your metadata.csv file here
metadata_path = 'path/to/metadata.csv'

# TODO: Replace with actual path to your TCT images directory here
image_base_dir = "/path/to/TCT-dataset/images/"

# Load model and processor
processor = LlavaNextProcessor.from_pretrained(MODEL_NAME)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Prepare output CSV
with open(file_name, 'w') as f:
    f.write("file_name,type,prompt,response\n")

df = pd.read_csv(metadata_path)

count = 1
for index, row in df.iterrows():
    print(f"\n=== Processing Image {count}: {row['file_name']} ===")
    
    image_path = image_base_dir + row['file_name']
    image = Image.open(image_path).convert('RGB')

    # Questions
    queries = [
        row['tsc_question'],
        row['tad_question'],
        row['btp_question'],
        row['tcap_question']
    ]
    types_list = ["tsc", "tad", "btp", "tcap"]

    conversation = []   # Full chat history

    for i in range(len(queries)):
        question = queries[i] + " Output your multiple-choice answer first (A, B, C, D, E, or F), then explain your reasoning in 2-3 sentences."

        if i == 0:
            user_message = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        else:
            user_message = {"role": "user", "content": [{"type": "text", "text": question}]}

        conversation.append(user_message)

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(prompt, image, return_tensors="pt").to("cuda")

        output = model.generate(
            **inputs, 
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0
        )

        # Decode and extract ONLY the latest assistant response
        full_decoded = processor.decode(output[0], skip_special_tokens=True)
        
        if "ASSISTANT:" in full_decoded:
            response = full_decoded.split("ASSISTANT:")[-1].strip()
        else:
            response = full_decoded.strip()

        with open(file_name, 'a', encoding='utf-8') as f:
            # Escape double quotes by doubling them (CSV standard)
            safe_prompt = queries[i].replace('"', '""')
            safe_response = response.replace('"', '""')
            
            f.write(f'{image_path},"{types_list[i]}","{safe_prompt}","{safe_response}"\n')

        print(f"[{types_list[i].upper()}] Response:\n{response}\n")

        # Add assistant's response to history for next turn
        conversation.append({
            "role": "assistant", 
            "content": [{"type": "text", "text": response}]
        })

    count += 1

print(f"\nFinished processing all images! Results saved to: {file_name}")