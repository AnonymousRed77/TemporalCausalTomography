import os
import torch
import pandas as pd
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ========================= CONFIG =========================
MODEL_NAME = "OpenGVLab/InternVL3_5-30B-A3B"

file_name = "internvl_3_5_30b_tct.csv"

# TODO: Replace with actual path to your metadata.csv file here
metadata_path = 'path/to/metadata.csv'

# TODO: Replace with actual path to your TCT images directory here
image_base_dir = "/path/to/TCT-dataset/images/"

MULTI_TURN = True  

# ========================= LOAD MODEL =========================
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map="auto"
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False
)

# ========================= IMAGE PREPROCESS =========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    def find_best_ratio():
        best_diff = float('inf')
        best = (1, 1)
        for ratio in target_ratios:
            diff = abs(aspect_ratio - ratio[0] / ratio[1])
            if diff < best_diff:
                best_diff = diff
                best = ratio
        return best

    ratio = find_best_ratio()
    target_width = 448 * ratio[0]
    target_height = 448 * ratio[1]

    resized = image.resize((target_width, target_height))
    blocks = ratio[0] * ratio[1]

    images = []
    for i in range(blocks):
        box = (
            (i % (target_width // 448)) * 448,
            (i // (target_width // 448)) * 448,
            ((i % (target_width // 448)) + 1) * 448,
            ((i // (target_width // 448)) + 1) * 448
        )
        images.append(resized.crop(box))

    if use_thumbnail and len(images) != 1:
        images.append(image.resize((448, 448)))

    transform = build_transform(448)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values

# ========================= OUTPUT FILE =========================
with open(file_name, "w", encoding="utf-8") as f:
    f.write("file_name,type,prompt,response\n")

df = pd.read_csv(metadata_path)

# ========================= MAIN LOOP =========================
for idx, row in df.iterrows():
    print(f"\n=== Processing {row['file_name']} ===")

    image_path = os.path.join(image_base_dir, row['file_name'])

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        continue

    pixel_values = dynamic_preprocess(image).to(torch.bfloat16).cuda()

    queries = [
        row["tsc_question"],
        row["tad_question"],
        row["btp_question"],
        row["tcap_question"]
    ]
    types_list = ["tsc", "tad", "btp", "tcap"]

    history = []  # InternVL conversation history

    for i in range(len(queries)):

        question = queries[i] + " Output your multiple-choice answer first (A, B, C, D, E, or F), then explain your reasoning in 2-3 sentences."

        prompt = "<image>\n" + question

        # Reset history if needed
        if not MULTI_TURN:
            history = []

        generation_config = dict(
            max_new_tokens=512,
            do_sample=False
        )

        # ========================= INFERENCE =========================
        response, history = model.chat(
            tokenizer,
            pixel_values,
            prompt,
            generation_config=generation_config,
            history=history,
            return_history=True
        )

        print(f"[{types_list[i].upper()}] {response}\n")

        # ========================= CSV WRITE =========================
        safe_prompt = queries[i].replace('"', '""')
        safe_response = response.replace('"', '""')

        with open(file_name, "a", encoding="utf-8") as f:
            f.write(f'{image_path},"{types_list[i]}","{safe_prompt}","{safe_response}"\n')

print(f"\nFinished! Saved to {file_name}")