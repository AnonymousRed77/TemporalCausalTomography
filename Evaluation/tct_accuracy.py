import pandas as pd
import re
import os

# ---------------------------
# Helper: extract first letter (A-F)
# ---------------------------
def extract_choice(text):
    if pd.isna(text):
        return None
    match = re.search(r'\b([A-F])\b', str(text))
    return match.group(1) if match else None


# ---------------------------
# Load metadata (ground truth)
# ---------------------------
# TODO: Replace with actual path to your metadata.csv file here
metadata = 'path/to/metadata.csv'

# Convert metadata into long format: one row per (file_name, type)
rows = []
for _, row in metadata.iterrows():
    fname = row["file_name"]
    band = row["temporal_scale_band"]

    rows.append((fname, "tsc", row["tsc_answer"], band))
    rows.append((fname, "tad", row["tad_answer"], band))
    rows.append((fname, "btp", row["btp_answer"], band))
    rows.append((fname, "tcap", row["tcap_answer"], band))

gt_df = pd.DataFrame(rows, columns=["file_name", "type", "gt", "temporal_scale_band"])


# ---------------------------
# Load model responses
# ---------------------------
# TODO: Replace with the path to your VLM result CSV file here
resp = pd.read_csv("/path/to/vlm/response/file.csv")

# Extract just filename (strip path)
resp["file_name"] = resp["file_name"].apply(lambda x: os.path.basename(x))

# Extract predicted choice
resp["pred"] = resp["response"].apply(extract_choice)

# Keep relevant columns
resp = resp[["file_name", "type", "pred"]]


# ---------------------------
# Merge predictions with GT
# ---------------------------
df = pd.merge(gt_df, resp, on=["file_name", "type"], how="inner")

# Drop rows where prediction failed
df = df.dropna(subset=["pred"])

# ---------------------------
# Compute correctness
# ---------------------------
df["correct"] = df["gt"] == df["pred"]


# ---------------------------
# Overall accuracy
# ---------------------------
overall_acc = df["correct"].mean()
print(f"Overall Accuracy: {overall_acc:.4f}")


# ---------------------------
# Accuracy per probe/type
# ---------------------------
probe_acc = df.groupby("type")["correct"].mean()
print("\nAccuracy per probe:")
print(probe_acc)


# ---------------------------
# Accuracy per temporal scale band
# ---------------------------
band_acc = df.groupby("temporal_scale_band")["correct"].mean()
print("\nAccuracy per temporal scale band:")
print(band_acc)