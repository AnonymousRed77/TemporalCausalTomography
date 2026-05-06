import os
import pandas as pd
from PIL import Image
import base64
import io
import re

from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen3.5-9B"

# TODO: Replace with actual path to your metadata.csv file here
metadata_path = 'path/to/metadata.csv'

# TODO: Replace with actual path to your TCT images directory here
image_base_dir = "/path/to/TCT-dataset/images/"

output_file = "qwen3_5_9b_tct.csv"

def pil_to_data_url(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_prompt(history, image_url, question):
    messages = history.copy()

    user_content = []

    if len(history) == 0:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })

    user_content.append({
        "type": "text",
        "text": question
    })

    messages.append({
        "role": "user",
        "content": user_content
    })

    return messages

def extract_final_answer(text: str) -> str:
    """
    Removes everything before </think> if present.
    Keeps only final answer.
    """
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    # optional cleanup
    return text.strip()

# ========================= MAIN =========================
if __name__ == "__main__":

    print("Loading vLLM model...")

    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.9
    )

    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=0.95, 
        top_k=20, 
        min_p=0.0, 
        presence_penalty=1.5, 
        repetition_penalty=1.0,
        max_tokens=50000
    )

    print("Model loaded.")

    df = pd.read_csv(metadata_path)

    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("file_name,type,prompt,response\n")

    for idx, row in df.iterrows():

        image_path = os.path.join(image_base_dir, row["file_name"])
        print(f"\n=== Processing {image_path} ===")

        image = Image.open(image_path).convert("RGB")
        image_url = pil_to_data_url(image)

        questions = [
            ("tsc", row["tsc_question"]),
            ("tad", row["tad_question"]),
            ("btp", row["btp_question"]),
            ("tcap", row["tcap_question"]),
        ]

        history = []

        for qtype, question in questions:

            prompt = question + " Output your multiple-choice answer first (A, B, C, D, E, or F), then explain your reasoning in 2-3 sentences."

            messages = build_prompt(history, image_url, prompt)

            response = "None"  # default if failure happens

            try:
                outputs = llm.chat(
                    messages=messages,
                    sampling_params=sampling_params
                )

                raw_response = outputs[0].outputs[0].text
                response = extract_final_answer(raw_response)

            except Exception as e:
                print(f"[{qtype}] ERROR: {e}")

            print(f"[{qtype}] {response}\n")

            with open(output_file, "a") as f:
                f.write(
                    f'"{image_path}","{qtype}","{question}","{response.replace(chr(34), chr(34)*2)}"\n'
                )

            if response != "None":
                history.append({
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                })

                history.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}]
                })

    print("Done.")
    