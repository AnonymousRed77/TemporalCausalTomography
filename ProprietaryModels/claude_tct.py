import os
import logging
import time
import anthropic
import json
import csv
import pandas as pd 
import base64

from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception
import httpx # Used by Anthropic client, good to import explicitly for error handling

# ========================= CONFIG =========================
raw_api_key = os.environ.get("ANTHROPIC_API_KEY")
if raw_api_key:
    anthropic_api_key = raw_api_key.strip()
else:
    anthropic_api_key = None # Explicitly set to None if env var is not set

if not anthropic_api_key:
    if raw_api_key == "": # If it was set but empty
        raise ValueError("ANTHROPIC_API_KEY environment variable is set but is empty. Please check its value.")
    else: # If it was not set at all
        raise ValueError("ANTHROPIC_API_KEY environment variable not set. Please ensure it's defined.")

# Initialize Anthropic client with the API key
client = anthropic.Anthropic(api_key=anthropic_api_key)

output_file_name = "claude_tct.csv" 

# TODO: Replace with actual path to your metadata.csv file here
metadata_path = 'path/to/metadata.csv'

# TODO: Replace with actual path to your TCT images directory here
image_base_dir = "/path/to/TCT-dataset/images/"

MODEL_NAME = "claude-sonnet-4-6" 

# ========================= LOGGING =========================
# Configure logging to INFO level for more visibility during execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ========================= OUTPUT FILE SETUP =========================
# Delete the output file if it already exists to prevent appending to old data
if os.path.exists(output_file_name):
    os.remove(output_file_name)
    logger.info(f"Removed existing output file: {output_file_name} for a fresh start.")

# Always write the header for the new file
with open(output_file_name, 'w', encoding='utf-8') as f:
    f.write("file_name,type,prompt,response\n")

# Load metadata for images and questions
df = pd.read_csv(metadata_path)


# ========================= UTILITY FUNCTIONS =========================

# Function to encode the image to base64
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


# ========================= RETRY MECHANISM =========================
class ClaudeAPIError(Exception):
    """Custom exception for Claude API errors."""
    pass

@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential_jitter(initial=2, max=60, jitter=5),
    retry=retry_if_exception(lambda e: (
        isinstance(e, anthropic.APIStatusError) and e.status_code in [429, 500, 502, 503, 504] # Rate limit, server errors
    ) or isinstance(e, httpx.RequestError) # Network errors
    or isinstance(e, ClaudeAPIError) # Catch general Claude API issues
    ),
    reraise=True # Re-raise the last exception if all retries fail
)
def generate_with_retry(messages_history, model_name=MODEL_NAME):
    """Tries to create a message, retrying on specific API/network errors."""
    try:
        response_message = client.messages.create(
            model=model_name,
            max_tokens=1024, 
            messages=messages_history
        )

        if response_message.content and len(response_message.content) > 0 and hasattr(response_message.content[0], 'text'):
            return response_message.content[0].text
        else:
            logger.error(f"Unexpected Claude API response structure: {response_message}")
            raise ClaudeAPIError(f"Unexpected Claude API response: {response_message}")

    except anthropic.APIStatusError as e:
        logger.warning(f"Claude API status error (will retry): Status {e.status_code} - {e.response} - {e.message}")
        raise ClaudeAPIError(f"Claude API status error: {e.status_code} - {e.message}")
    except httpx.RequestError as e:
        logger.warning(f"Network request error (will retry): {e}")
        raise ClaudeAPIError(f"Network error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during API call: {e}")
        raise ClaudeAPIError(f"Unexpected error: {e}")


# ========================= MAIN LOOP =========================
global_image_count = 1
for index, row in df.iterrows():
    print(f"\n=== Image {global_image_count}: {row['file_name']} ===")

    image_path = os.path.join(image_base_dir, row['file_name'])

    try:
        base64_image = encode_image(image_path)
    except Exception as e:
        logger.error(f"Failed to load or encode image: {image_path} - {e}")
        continue

    # TCT questions from the metadata
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

        # ========================= BUILD USER TURN =========================
        user_content_blocks = []
        if i == 0: # Attach image only for the first turn in the conversation
            user_content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg", # Assuming JPEG, adjust if other types are present
                    "data": base64_image
                }
            })
        user_content_blocks.append({"type": "text", "text": question})

        messages.append({"role": "user", "content": user_content_blocks})

        response_text = "" # Initialize response text
        try:
            # Call Claude with retry mechanism
            raw_claude_response = generate_with_retry(messages_history=messages, model_name=MODEL_NAME)
            response_text = raw_claude_response if raw_claude_response else ""

            print(f"[{types_list[i].upper()}] {response_text}\n")

            # ========================= SAVE =========================
            safe_prompt = queries[i].replace('"', '""')
            safe_response = response_text.replace('"', '""')

            with open(output_file_name, 'a', encoding='utf-8') as f:
                f.write(f'"{image_path}","{types_list[i]}","{safe_prompt}","{safe_response}"\n')

            messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})

        except Exception as e:
            logger.error(f"Failed after retries for {image_path}, type {types_list[i]}: {e}")
            error_msg = str(e).replace('"', '""')

            with open(output_file_name, 'a', encoding='utf-8') as f:
                f.write(f'"{image_path}","{types_list[i]}","{safe_prompt}","ERROR: {error_msg}"\n')

            # Add an error message to conversation history to maintain structure
            messages.append({"role": "assistant", "content": [{"type": "text", "text": f"Error during generation: {error_msg}"}]})


        time.sleep(1.5)

    global_image_count += 1

print(f"\nFinished! Saved to {output_file_name}")