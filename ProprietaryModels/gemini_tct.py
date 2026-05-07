from google import genai
from google.genai import types
import os
import logging
import time
import pandas as pd

from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception
import google.genai.errors as genai_errors

# ========================= CONFIG =========================
file_name = "gemini25pro_tct.csv"

# TODO: Replace with actual path to your metadata.csv file here
metadata_path = 'path/to/metadata.csv'

# TODO: Replace with actual path to your TCT images directory here
image_base_dir = "/path/to/TCT-dataset/images/"

MODEL_NAME = "gemini-2.5-pro"

# ========================= LOGGING =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================= OUTPUT FILE =========================
if os.path.exists(file_name):
    os.remove(file_name)
    logger.info(f"Removed existing output file: {file_name} for a fresh start.")

with open(file_name, 'w', encoding='utf-8') as f:
    f.write("file_name,type,prompt,response\n")

df = pd.read_csv(metadata_path)

# ========================= RETRY =========================
@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential_jitter(initial=2, max=60, jitter=5),
    retry=retry_if_exception(lambda e: (
        isinstance(e, genai_errors.ServerError) and getattr(e, 'status_code', 0) == 503
    ) or isinstance(e, (genai_errors.ServerError, genai_errors.APIError)) or
        (hasattr(e, 'code') and e.code in (503, 429, 500, 502, 504))
    ),
)
def generate_with_retry(client, model, contents):
    return client.models.generate_content(
        model=model,
        contents=contents
    )

# ========================= MAIN LOOP =========================
START_IMAGE_NUMBER = 1
# Calculate the DataFrame index to start from (0-based)
START_DF_INDEX = START_IMAGE_NUMBER - 1

# Initialize client outside the loop
client = genai.Client()

for index, row in df.iloc[START_DF_INDEX:].iterrows():
    current_image_count = index + 1 

    print(f"\n=== Image {current_image_count}: {row['file_name']} ===")

    image_path = os.path.join(image_base_dir, row['file_name'])

    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        continue

    # TCT questions
    queries = [
        row['tsc_question'],
        row['tad_question'],
        row['btp_question'],
        row['tcap_question']
    ]
    types_list = ["tsc", "tad", "btp", "tcap"]

    # Conversation history
    contents = []

    for i in range(len(queries)):

        question = queries[i] + (
            " Output your multiple-choice answer first (A, B, C, D, E, or F), "
            "then explain your reasoning in 2-3 sentences."
        )

        # ========================= BUILD TURN =========================
        if i == 0:
            contents.append({
                "role": "user",
                "parts": [
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/jpeg'
                    ),
                    types.Part.from_text(text=question)
                ]
            })
        else:
            contents.append({
                "role": "user",
                "parts": [types.Part.from_text(text=question)]
            })

        try:
            output = generate_with_retry(
                client=client,
                model=MODEL_NAME,
                contents=contents
            )

            response = output.text if output.text else ""

            print(f"[{types_list[i].upper()}] {response}\n")

            # ========================= SAVE =========================
            safe_prompt = queries[i].replace('"', '""')
            safe_response = response.replace('"', '""')

            with open(file_name, 'a', encoding='utf-8') as f:
                f.write(f'"{image_path}","{types_list[i]}","{safe_prompt}","{safe_response}"\n')

            contents.append({
                "role": "model",
                "parts": [types.Part.from_text(text=response)]
            })

        except Exception as e:
            logger.error(f"Failed after retries: {e}")
            error_msg = str(e).replace('"', '""')

            with open(file_name, 'a', encoding='utf-8') as f:
                f.write(f'"{image_path}","{types_list[i]}","{safe_prompt}","ERROR: {error_msg}"\n')

        time.sleep(1.5)

print(f"\Finished! Saved to {file_name}")