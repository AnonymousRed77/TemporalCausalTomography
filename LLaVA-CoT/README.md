# LLaVA-CoT

Follow the following steps to set up the enviromment and run the inference for LLaVA-CoT. 

## Minimum configuration

We used 1 NVIDIA A40 GPU with 48GB to run and fine-tune this model.

## Getting Started

1. Create a dirctory and name it ``LLaVA-CoT``

2. Go to the ``LLaVA-CoT`` directory
   
``cd LLaVA-CoT``

3. Download the ``llava_cot.yml`` file and move it to the ``LLaVA-CoT`` directory

4. Create and activate the environment

``conda env create -f llava_cot.yml``

``conda activate llava_cot``

## Inference

1. Download the ``llava_cot_tct.py`` script and move it to ``LLaVA-CoT``

2. Open ``llava_cot_tct.py`` and modify the code under 2 TODOs in this script

- Line 14: Replace with actual path to your metadata.csv file (downloaded from our dataset)
- Line 17: Replace with actual path to your TCT-dataset/images directory (downloaded from our dataset)

3. Save the code, go to ``LLaVA-CoT`` and run the code there

``python llava_cot_tct.py``

This will then create a CSV file that contains the image file paths, the probe types (tsc, tad, btp, and tcap), the questions, and LLaVA-CoT's responses to these questions.
