import pandas as pd
import numpy as np
import os
from sklearn.utils import resample
from scipy.stats import pearsonr, spearmanr

def evaluate_model(metadata_path, results_path):
    """
    Evaluates a single model file. 
    Returns: Accuracy, Overall TCI, TCI Decomposed, and CI bounds.
    """
    df_meta = pd.read_csv(metadata_path)
    df_results = pd.read_csv(results_path)

    # 1. Pre-process Metadata
    gt_map = {}
    for _, row in df_meta.iterrows():
        gt_map[row['file_name']] = {
            'tsc': str(row['tsc_answer']).strip(),
            'tad': str(row['tad_answer']).strip(),
            'btp': str(row['btp_answer']).strip(),
            'tcap': str(row['tcap_answer']).strip()
        }

    # 2. Pre-process Model Results & Calculate Standard Accuracy
    model_responses = {}
    correct_count = 0
    total_questions = 0

    for _, row in df_results.iterrows():
        clean_fname = os.path.basename(row['file_name']).strip('"')
        raw_response = str(row['response']).strip()
        choice = raw_response[0].upper() if raw_response else None
        p_type = row['type'].lower()

        if clean_fname not in model_responses:
            model_responses[clean_fname] = {}
        model_responses[clean_fname][p_type] = choice

        # Calculate standard MCQ accuracy
        if clean_fname in gt_map and p_type in gt_map[clean_fname]:
            total_questions += 1
            if choice == gt_map[clean_fname][p_type]:
                correct_count += 1

    model_accuracy = correct_count / total_questions if total_questions > 0 else 0

    # 3. Calculate Image-level TCI and Decomposition
    constraint_pairs = [
        ('tsc', 'tad'), ('tsc', 'btp'), ('tad', 'btp'), ('tsc', 'tcap')
    ]
    
    image_tci_scores = []
    # Dictionary to track success per pair across all images
    pair_results = {pair: [] for pair in constraint_pairs}

    for img_id, gt in gt_map.items():
        if img_id not in model_responses: continue
        m_choices = model_responses[img_id]
        
        img_satisfied = 0
        img_applicable = 0

        for p1, p2 in constraint_pairs:
            if p1 in gt and p2 in gt and p1 in m_choices and p2 in m_choices:
                if pd.notna(gt[p1]) and pd.notna(gt[p2]):
                    img_applicable += 1
                    # A pair is consistent if both are correct
                    consistent = 1 if (m_choices[p1] == gt[p1] and m_choices[p2] == gt[p2]) else 0
                    img_satisfied += consistent
                    pair_results[(p1, p2)].append(consistent)
        
        if img_applicable > 0:
            image_tci_scores.append(img_satisfied / img_applicable)

    # 4. Aggregate TCI Metrics
    headline_tci = np.mean(image_tci_scores)
    
    # Decomposed Scores
    decomp = {f"{p1.upper()}-{p2.upper()}": np.mean(res) for (p1, p2), res in pair_results.items() if res}

    # Bootstrap for CI
    bootstrapped_means = [np.mean(resample(image_tci_scores)) for _ in range(1000)]
    lower_ci = np.percentile(bootstrapped_means, 2.5)
    upper_ci = np.percentile(bootstrapped_means, 97.5)

    return model_accuracy, headline_tci, decomp, (lower_ci, upper_ci)

# --- Main Analysis Loop ---

# 1. Define your models and paths
# TODO: Replace these with the actual paths to all VLM result files
model_files = {
    "LLaVA-NeXT-13B": "/path/to/vlm/response/file.csv",
    "LLaVA-CoT-11B": "/path/to/vlm/response/file.csv",
    "InternVL3.5-8B": "/path/to/vlm/response/file.csv",
    "Qwen3-VL-8B": "/path/to/vlm/response/file.csv",
    "Phi4-Reasoning-15B": "/path/to/vlm/response/file.csv",
    "GPT-4o": "/path/to/vlm/response/file.csv",
    "Claude-4.6-Sonnet": "/path/to/vlm/response/file.csv",
    "Gemini-2.5-Pro": "/path/to/vlm/response/file.csv",
    "Qwen3.5-9B": "/path/to/vlm/response/file.csv",
    "InternVL3.5-30B": "/path/to/vlm/response/file.csv",
    "Qwen3-VL-30B": "/path/to/vlm/response/file.csv"
}

# TODO: Replace with actual path to your metadata.csv file here
metadata_csv = 'path/to/metadata.csv'

all_stats = []

print(f"{'Model':<20} | {'Accuracy':<10} | {'TCI':<10} | {'95% CI'}")
print("-" * 60)

for name, path in model_files.items():
    acc, tci, decomp, ci = evaluate_model(metadata_csv, path)
    
    all_stats.append({
        'model': name,
        'accuracy': acc,
        'tci': tci,
        **decomp
    })
    
    print(f"{name:<20} | {acc:.4f}     | {tci:.4f}    | [{ci[0]:.3f}, {ci[1]:.3f}]")

# 2. Print Decomposition Table
df_stats = pd.DataFrame(all_stats)
print("\n=== TCI CONSTRAINT-PAIR DECOMPOSITION ===")
decomp_cols = ['model', 'TSC-TAD', 'TSC-BTP', 'TAD-BTP', 'TSC-TCAP']
print(df_stats[decomp_cols].to_string(index=False))

# 3. Correlation Analysis
if len(all_stats) >= 2:
    acc_vals = df_stats['accuracy']
    tci_vals = df_stats['tci']
    
    p_r, p_p = pearsonr(acc_vals, tci_vals)
    s_r, s_p = spearmanr(acc_vals, tci_vals)
    
    print("\n=== CORRELATION ANALYSIS (Accuracy vs. TCI) ===")
    print(f"Pearson r:  {p_r:.4f} (p={p_p:.4e})")
    print(f"Spearman ρ: {s_r:.4f} (p={s_p:.4e})")
else:
    print("\n[Need at least 2 models to compute correlation]")