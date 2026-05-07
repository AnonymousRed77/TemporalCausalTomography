import pandas as pd
import numpy as np
import os
from statsmodels.stats.inter_rater import fleiss_kappa

def calculate_tad_metrics_with_kappa(metadata_path, model_response_path, judge_csv_list):
    # 1. Load Metadata
    meta_df = pd.read_csv(metadata_path)
    meta_df['file_name'] = meta_df['file_name'].apply(lambda x: os.path.basename(str(x)).strip('"'))
    
    # 2. MCQ Accuracy (Fluency)
    model_df = pd.read_csv(model_response_path)
    model_df['file_name'] = model_df['file_name'].apply(lambda x: os.path.basename(str(x)).strip('"'))
    model_df = model_df[model_df['type'].str.lower() == 'tad']
    
    merged_mcq = pd.merge(model_df, meta_df[['file_name', 'tad_answer']], on='file_name')
    merged_mcq['model_choice'] = merged_mcq['response'].astype(str).str.strip().str[0].str.upper()
    gt_series = merged_mcq['tad_answer'].astype(str).str.strip().str.upper()
    merged_mcq['is_correct'] = (merged_mcq['model_choice'] == gt_series).astype(int)
    mcq_accuracy = merged_mcq['is_correct'].mean()

    # 3. Process Judge Data
    judge_data = []
    for i, path in enumerate(judge_csv_list):
        df = pd.read_csv(path)
        df['file_name'] = df['file_name'].apply(lambda x: os.path.basename(str(x)).strip('"'))
        df['judge_id'] = f"judge_{i+1}"
        judge_data.append(df)
    combined_judges = pd.concat(judge_data)

    # --- FLEISS' KAPPA (Agreement) ---
    def compute_dimension_kappa(df, dimension_col):
        # Drop rows missing ANY judge to ensure a balanced matrix
        pivot_df = df.pivot(index='file_name', columns='judge_id', values=dimension_col).dropna()
        # Filter out 'C' to measure agreement on logic, not formatting
        pivot_df = pivot_df[~(pivot_df == 'C').any(axis=1)]

        # ADD THIS DEBUG LINE:
        print(f"DEBUG: {dimension_col} Kappa calculated on {len(pivot_df)} images.")
        
        if pivot_df.empty: return 1.0
        
        counts_matrix = np.array([[(row == 'A').sum(), (row == 'B').sum()] for _, row in pivot_df.iterrows()])
        
        if np.all(counts_matrix == counts_matrix[0, :]): return 1.0
            
        try:
            k = fleiss_kappa(counts_matrix)
            return 1.0 if np.isnan(k) else k
        except:
            return 1.0

    kappas = {dim: compute_dimension_kappa(combined_judges, col) 
              for dim, col in [("CTF", "ctf"), ("CG", "cg"), ("RC", "rc")]}

    # --- MAJORITY VOTE (Order-Independent) ---
    def safe_idxmax(x):
        counts = x.value_counts()
        if counts.empty: return 'B'
        if counts.max() < 2: return 'B' # 1-1-1 tie results in Failure
        return counts.idxmax()

    majority_votes = combined_judges.groupby('file_name').agg({
        'ctf': safe_idxmax, 'cg': safe_idxmax, 'rc': safe_idxmax
    }).reset_index()

    # 4. Final Metric Computation
    final_df = pd.merge(majority_votes, meta_df[['file_name', 'tad_answer']], on='file_name')
    faithfulness_rate = (final_df['ctf'] == 'A').mean()
    alignment_rate = (final_df['rc'] == 'A').mean()
    impossible_items = final_df[final_df['tad_answer'].str.strip().str.upper() == 'B']
    cg_rate = (impossible_items['cg'] == 'A').mean() if not impossible_items.empty else 0
    ffg = mcq_accuracy - faithfulness_rate

    # 5. Output
    print(f"\nRESULTS FOR: {os.path.basename(model_response_path)}")
    print("="*60)
    print(f"MCQ Accuracy (Fluency):          {mcq_accuracy:.4f}")
    print(f"Faithfulness Rate (CTF):         {faithfulness_rate:.4f}")
    print(f"Constraint Grounding Rate (CG):  {cg_rate:.4f}")
    print(f"Alignment Rate (RC):             {alignment_rate:.4f}")
    print(f"Fluency-Faithfulness Gap (FFG):  {ffg:.4f}")
    print("-" * 60)
    for dim, k in kappas.items():
        print(f" - {dim:<4} Kappa: {k:.4f}")
    print("="*60)

# --- RUN ANALYSIS ---

# TODO: Replace with actual path to your metadata.csv file here
metadata = 'path/to/metadata.csv'

# TODO: Replace with the path to your VLM result CSV file here
model_results = "/path/to/vlm/response/file.csv" 

# TODO: Replace with the three judge evaluation CSV files here
judges = ["/path/to/judge1/evaluation/file.csv", "/path/to/judge2/evaluation/file.csv", "/path/to/judge3/evaluation/file.csv"]

calculate_tad_metrics_with_kappa(metadata, model_results, judges)