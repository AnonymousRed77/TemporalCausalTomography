import pandas as pd
import numpy as np
import os
import re

def get_claimed_rank(question_text):
    """
    Advanced NLP logic to determine the Rank (1-5) of the claim in the question.
    """
    q = str(question_text).lower()
    
    # 1. Regex to find numbers and the word immediately following them
    num_matches = re.findall(r'(\d+)\s*([a-z]+)', q)
    
    # --- RANK 5: Decades to Millennia ---
    if any(w in q for w in ['millennia', 'thousand', 'million', 'billion', 'ancient', 'centuries', 'decades']):
        return 5
    if 'hundreds' in q and 'year' in q:
        return 5
    
    # --- Numerical Logic for Exceptions ---
    for num_str, unit in num_matches:
        val = int(num_str)
        if 'second' in unit and val >= 3600: return 2
        if 'minute' in unit and val >= 60: return 2
        if 'hour' in unit and val >= 168: return 3
        if 'day' in unit:
            if val >= 365: return 4 
            if val >= 30: return 3  
        if 'week' in unit and val >= 52: return 4

    # --- RANK 4: Years to Decades ---
    if any(w in q for w in ['long period', 'over time', 'seasons']):
        return 4
    if 'year' in q: return 4
        
    # --- RANK 3: Weeks to Months ---
    if any(w in q for w in ['month', 'week']): return 3
        
    # --- RANK 2: Hours to Days ---
    rank2_phrases = ['morning', 'afternoon', 'evening', 'today', 'tomorrow', 'yesterday', 'overnight', 'weekend', 'weekday', 'tonight']
    if any(w in q for w in rank2_phrases) or any(w in q for w in ['hour', 'day']):
        return 2
        
    # --- RANK 1: Seconds to Minutes ---
    if any(w in q for w in ['second', 'minute']): return 1
        
    return None 

def generate_tct_splits(metadata_path, results_path):
    df_meta = pd.read_csv(metadata_path)
    df_results = pd.read_csv(results_path)
    
    df_results['file_name'] = df_results['file_name'].apply(lambda x: os.path.basename(x).strip('"'))
    df = pd.merge(df_results, df_meta, on='file_name')

    # Scoring
    df['model_choice'] = df['response'].apply(lambda x: str(x).strip()[0].upper() if pd.notna(x) and len(str(x).strip())>0 else None)
    
    def get_gt(row):
        p = row['type'].lower()
        return str(row[f'{p}_answer']).strip() if f'{p}_answer' in row else None
    
    df['ground_truth'] = df.apply(get_gt, axis=1)
    df['is_correct'] = (df['model_choice'] == df['ground_truth']).astype(int)

    # TAD Difficulty Logic with Sub-Tier splitting
    def categorize_tad_tier(row):
        if row['type'].lower() != 'tad': return None
        
        scale_order = {'seconds_minutes': 1, 'hours_days': 2, 'weeks_months': 3, 'months_years': 4, 'decades_millennia': 5}
        actual_rank = scale_order.get(row['temporal_scale_band'], 3)
        claimed_rank = get_claimed_rank(row['tad_question'])
        
        # Determine base tier
        tier = 'Medium' # Fallback
        if claimed_rank is not None:
            distance = abs(actual_rank - claimed_rank)
            if distance >= 3: tier = 'Easy'
            elif distance == 2: tier = 'Medium'
            else: tier = 'Hard'
        
        # Force Plausible claims (A) into Hard
        if row['tad_answer'] == 'A':
            tier = 'Hard'

        # Create sub-tier for Hard
        if tier == 'Hard':
            if row['tad_answer'] == 'A':
                return 'Hard-Plausible'
            else:
                return 'Hard-Impossible'
        
        return tier

    df['tad_tier'] = df.apply(categorize_tad_tier, axis=1)

    # --- CALCULATE COUNTS ---
    tad_df = df[df['type'] == 'tad']
    tad_tier_dist = tad_df['tad_tier'].value_counts()
    tad_label_dist = tad_df['tad_answer'].value_counts()

    # BTP Direction Logic
    df['btp_direction'] = df.apply(lambda r: 'Backward' if r['type']=='btp' and any(w in str(r['btp_question']).lower() for w in ['before', 'ago', 'prior', 'past']) else ('Forward' if r['type']=='btp' else None), axis=1)

    # Generate Tables
    model_col = 'model' if 'model' in df.columns else None

    btp_table = pd.pivot_table(df[df['type'] == 'btp'], values='is_correct', index=model_col if model_col else 'type', columns='btp_direction', aggfunc='mean')
    
    tad_table = pd.pivot_table(df[df['type'] == 'tad'], values='is_correct', index=model_col if model_col else 'type', columns='tad_tier', aggfunc='mean')
    # Reorder columns to show the new split
    cols_order = [c for c in ['Easy', 'Medium', 'Hard-Plausible', 'Hard-Impossible'] if c in tad_table.columns]
    tad_table = tad_table.reindex(columns=cols_order)

    return btp_table, tad_table, tad_tier_dist, tad_label_dist

# --- Execution ---
try:
    # TODO: Replace with actual path to your metadata.csv file here
    meta_p = 'path/to/metadata.csv'
    
    # TODO: Replace with the path to your VLM result CSV file here
    res_p = pd.read_csv("/path/to/vlm/response/file.csv")
    
    btp_res, tad_res, t_tiers, t_labels = generate_tct_splits(meta_p, res_p)
    
    print("\n" + "="*50)
    print("TAD DATASET COMPOSITION (SPLIT TIERS)")
    print("="*50)
    for tier in ['Easy', 'Medium', 'Hard-Plausible', 'Hard-Impossible']:
        print(f" - {tier:15}: {t_tiers.get(tier, 0)} items")
    
    print("\n=== TABLE 1: BTP Asymmetry (Accuracy) ===\n", btp_res)
    print("\n=== TABLE 2: TAD Accuracy by Tier (Hard-Split) ===\n", tad_res)
    
except Exception as e:
    import traceback
    traceback.print_exc()