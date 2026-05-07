## Evaluation

Evaluation proceeds in two empirical tiers. Tier 1 performs exact-match scoring on probe outputs and computes metrics from probe choices alone, with no LLM-judge dependence. Tier 2 supplements Tier 1 with LLM-judge analyses of textual explanations on TAD items.

## Getting Started

1. Create a dirctory and name it ``TCT-Eval``

2. Go to the ``TCT-Eval`` directory
   
``cd TCT-Eval``

## Tier 1 Evaluation

### Accuracy

1. Download the ``tct_accuracy.py`` script and move it to ``TCT-Eval``

2. Open ``tct_accuracy.py`` and modify the code under 2 TODOs in this script

- Line 19: Replace with actual path to your metadata.csv file (downloaded from our dataset)
- Line 39: Replace with actual path to your VLM result CSV file

3. Save the code, go to ``TCT-Eval`` and run the code there

``python tct_accuracy.py``

This will then print out the Overal Accuracy, Per-Probe Accuracy, and Per-Band Accuracy of the model being evaluated.

### Temporal Coherence Index (TCI) and TCI Constraint-Pair Decomposition 

1. Download the ``tct_tci.py`` script and move it to ``TCT-Eval``

2. Open ``tct_tci.py`` and modify the code under 2 TODOs in this script

- Lines 94-104: Replace with the actual paths to all VLM result files
- Line 108: Replace with actual path to your metadata.csv file (downloaded from our dataset)

3. Save the code, go to ``TCT-Eval`` and run the code there

``python tct_tci.py``

This will then print out a table of the Overall Accuracy and TCI values with 95% confidence intervals, and a table of TCI Constraint-Pair Decomposition of the models being evaluated.

### BTP and TAD Accuracy Breakdown

1. Download the ``tct_btp_tad.py`` script and move it to ``TCT-Eval``

2. Open ``tct_btp_tad.py`` and modify the code under 2 TODOs in this script

- Line 121: Replace with actual path to your metadata.csv file (downloaded from our dataset)
- Line 124: Replace with actual path to your VLM result CSV file

3. Save the code, go to ``TCT-Eval`` and run the code there

``python tct_btp_tad.py``

This will then print out the BTP Forward and Backward Accuracy and the TAD Accuracy by Difficulty Tier of the model being evaluated.

## Tier 2 Evaluation

### Independent LLM Judge Evaluation

1. Download the ``proprietary_models.yml`` file from ``ProprietaryModels`` and move it to the ``TCT-Eval`` directory

2. Create and activate the environment

``conda env create -f proprietary_models.yml``

``conda activate proprietary_models``

### GPT-4o Evaluation

1. Download the ``gpt4_tct_tad_eval.py`` script and move it to ``TCT-Eval``

2. Open ``gpt4_tct_tad_eval.py`` and modify the code under 3 TODOs in this script

- Line 14: Replace with actual path to your metadata.csv file (downloaded from our dataset)
- Line 17: Replace with actual path to your VLM result CSV file
- Line 20: Replace 'model_name' with the VLM being evaluated (e.g., llava_next, llava_cot, qwen3vl, etc.)

3. Create an OpenAI API key and set it as an environment variable using the following command

``export OPENAI_API_KEY='YOUR-API-KEY'``

4. Save the code, go to ``TCT-Eval`` and run the code there

``python gpt4_tct_tad_eval.py``

This will then create a CSV file that contains the image file paths and the judge's evaluation to the VLM's TAD responses.

**Warning:** Do not evaluate GPT-4o with this judge to mitigate the self-preference biases.

### Claude 4.6 Sonnet Evaluation

1. Download the ``claude_tct_tad_eval.py`` script and move it to ``TCT-Eval``

2. Open ``claude_tct_tad_eval.py`` and modify the code under 3 TODOs in this script

- Line 19: Replace with actual path to your metadata.csv file (downloaded from our dataset)
- Line 22: Replace with actual path to your VLM result CSV file
- Line 25: Replace 'model_name' with the VLM being evaluated (e.g., llava_next, llava_cot, qwen3vl, etc.)

3. Create an Anthropic API key and set it as an environment variable using the following command

``export ANTHROPIC_API_KEY='YOUR-API-KEY'``

4. Save the code, go to ``TCT-Eval`` and run the code there

``python claude_tct_tad_eval.py``

This will then create a CSV file that contains the image file paths and the judge's evaluation to the VLM's TAD responses.

**Warning:** Do not evaluate Claude 4.6 Sonnet with this judge to mitigate the self-preference biases.

### Gemini 2.5 Pro Evaluation

1. Download the ``gemini_tct_tad_eval.py`` script and move it to ``TCT-Eval``

2. Open ``gemini_tct_tad_eval.py`` and modify the code under 3 TODOs in this script

- Line 20: Replace with actual path to your metadata.csv file (downloaded from our dataset)
- Line 23: Replace with actual path to your VLM result CSV file
- Line 26: Replace 'model_name' with the VLM being evaluated (e.g., llava_next, llava_cot, qwen3vl, etc.)

3. Create an Gemini API key and set it as an environment variable using the following command

``export GEMINI_API_KEY='YOUR-API-KEY'``

4. Save the code, go to ``TCT-Eval`` and run the code there

``python gemini_tct_tad_eval.py``

This will then create a CSV file that contains the image file paths and the judge's evaluation to the VLM's TAD responses.

**Warning:** Do not evaluate Gemini 2.5 Pro with this judge to mitigate the self-preference biases.

### Grok 4.20 Evaluation

1. Download the ``grok4_tct_tad_eval.py`` script and move it to ``TCT-Eval``

2. Open ``grok4_tct_tad_eval.py`` and modify the code under 3 TODOs in this script

- Line 22: Replace with actual path to your metadata.csv file (downloaded from our dataset)
- Line 25: Replace with actual path to your VLM result CSV file
- Line 28: Replace 'model_name' with the VLM being evaluated (e.g., llava_next, llava_cot, qwen3vl, etc.)

3. Create an XAI API key and set it as an environment variable using the following command

``export XAI_API_KEY='YOUR-API-KEY'``

4. Save the code, go to ``TCT-Eval`` and run the code there

``python grok4_tct_tad_eval.py``

This will then create a CSV file that contains the image file paths and the judge's evaluation to the VLM's TAD responses.

**Warning:** Only evaluate GPT-4o, Claude 4.6 Sonnet, and Gemini 2.5 Pro with this judge.

### Majority-Voting Judge Evaluation

1. Download the ``tct_tad_judge_analysis.py`` script and move it to ``TCT-Eval``

2. Open ``tct_tad_judge_analysis.py`` and modify the code under 3 TODOs in this script

- Line 91: Replace with actual path to your metadata.csv file (downloaded from our dataset)
- Line 94: Replace with actual path to your VLM result CSV file
- Line 97: Replace with the three judge evaluation CSV files

3. Save the code, go to ``TCT-Eval`` and run the code there

``python tct_tad_judge_analysis.py``

This will then print out the MCQ Accuracy, Faithfulness Rate (CTF), Constraint Grounding Rate (CG), Alignment Rate (RC), and Fluency-Faithfulness Gap (FFG) along with the Kappa values of CTF, CG, and RC of the model being evaluated.
