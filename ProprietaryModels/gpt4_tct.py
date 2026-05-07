import time
import os
import base64
import requests
import json
import csv
import logging
import pandas as pd

from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception

# ========================= CONFIG =========================
raw_api_key = os.environ.get("OPENAI_API_KEY")
if raw_api_key:
    api_key = raw_api_key.strip()
else:
    api_key = None # Explicitly set to None if env var is not set

if not api_key:
    if raw_api_key == "": # If it was set but empty
        raise ValueError("OPENAI_API_KEY environment variable is set but is empty. Please check its value.")
    else: # If it was not set at all
        raise ValueError("OPENAI_API_KEY environment variable not set. Please ensure it's defined.")


output_file_name = "gpt4_tct.csv"

# TODO: Replace with actual path to your metadata.csv file here
metadata_path = 'path/to/metadata.csv'

# TODO: Replace with actual path to your TCT images directory here
image_base_dir = "/path/to/TCT-dataset/images/"

MODEL_NAME = "gpt-4o"

# ========================= LOGGING =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================= OUTPUT FILE SETUP =========================
if os.path.exists(output_file_name):
    os.remove(output_file_name)
    logger.info(f"Removed existing output file: {output_file_name} for a fresh start.")

with open(output_file_name, 'w', encoding='utf-8') as f:
    f.write("file_name,type,prompt,response\n")

df = pd.read_csv(metadata_path)

# ========================= UTILITY FUNCTIONS =========================

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Headers for OpenAI API requests
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
# ADDED DEBUGGING
logger.info(f"API Key successfully loaded (first 5 chars): {api_key[:5]}...")
logger.info(f"Constructed Authorization header (first 20 chars): {headers['Authorization'][:20]}...")


# ========================= RETRY MECHANISM =========================
class OpenAIAPIError(Exception):
    pass

@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential_jitter(initial=2, max=60, jitter=5),
    retry=retry_if_exception(lambda e: (
        isinstance(e, requests.exceptions.RequestException) or
        isinstance(e, OpenAIAPIError)
    )),
    reraise=True
)
def generate_with_retry(messages_history, model_name=MODEL_NAME):
    payload = {
        "model": model_name,
        "messages": messages_history,
        "max_tokens": 1024
    }

    try:
        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        res.raise_for_status()

        response_json = res.json()
        if 'choices' in response_json and len(response_json['choices']) > 0:
            return response_json['choices'][0]['message']['content']
        else:
            logger.error(f"Unexpected API response structure: {response_json}")
            raise OpenAIAPIError(f"Unexpected API response: {response_json}")

    except requests.exceptions.RequestException as e:
        logger.warning(f"OpenAI API request failed (will retry): {e}. Response: {e.response.text if e.response else 'No response body'}")
        raise OpenAIAPIError(f"Request failed: {e}")
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error from OpenAI API (will retry): {e}, Raw response: {res.text if res else 'No response'}")
        raise OpenAIAPIError(f"JSON decode error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during API call: {e}")
        raise OpenAIAPIError(f"Unexpected error: {e}")


# ========================= MAIN LOOP =========================
count = 1
for index, row in df.iterrows():
    print(f"\n=== Image {count}: {row['file_name']} ===")

    image_path = os.path.join(image_base_dir, row['file_name'])

    try:
        base64_image = encode_image(image_path)
    except Exception as e:
        logger.error(f"Failed to load or encode image: {image_path} - {e}")
        continue

    queries = [
        row['tsc_question'],
        row['tad_question'],
        row['btp_question'],
        row['tcap_question']
    ]
    types_list = ["tsc", "tad", "btp", "tcap"]

    messages = []

    for i in range(len(queries)):
        question = queries[i] + (
            " Output your multiple-choice answer first (A, B, C, D, E, or F), "
            "then explain your reasoning in 2-3 sentences."
        )

        user_content = []
        if i == 0:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        user_content.append({"type": "text", "text": question})

        messages.append({"role": "user", "content": user_content})

        response = ""
        try:
            raw_response_text = generate_with_retry(messages_history=messages, model_name=MODEL_NAME)
            response = raw_response_text if raw_response_text else ""
            print(f"[{types_list[i].upper()}] {response}\n")

            safe_prompt = queries[i].replace('"', '""')
            safe_response = response.replace('"', '""')

            with open(output_file_name, 'a', encoding='utf-8') as f:
                f.write(f'"{image_path}","{types_list[i]}","{safe_prompt}","{safe_response}"\n')

            messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

        except Exception as e:
            logger.error(f"Failed after retries for {image_path}, type {types_list[i]}: {e}")
            error_msg = str(e).replace('"', '""')

            with open(output_file_name, 'a', encoding='utf-8') as f:
                f.write(f'"{image_path}","{types_list[i]}","{safe_prompt}","ERROR: {error_msg}"\n')

            messages.append({"role": "assistant", "content": [{"type": "text", "text": f"Error during generation: {error_msg}"}]})

        time.sleep(1.5)

    count += 1

print(f"\nFinished! Saved to {output_file_name}")