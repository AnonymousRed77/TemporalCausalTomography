# Qwen3.5

Follow the following steps to set up the enviromment and run the inference for Qwen3.5. 

## Minimum configuration

We used 1 NVIDIA A40 GPU with 48GB to run this model.

## Getting Started

1. Create a dirctory and name it ``Qwen3_5``

2. Go to the ``Qwen3_5`` directory
   
``cd Qwen3_5``

3. Download the ``qwen35.yml`` file and move it to the ``Qwen3_5`` directory

4. Create and activate the environment

``conda env create -f qwen35.yml``

``conda activate qwen35.yml``

## Inference

1. Download the ``qwen3_5_9b_tct.py`` script and move it to ``Qwen3_5``

2. Open ``qwen3_5_9b_tct.py`` and modify the code under 2 TODOs in this script

- Line 13: Replace with actual path to your metadata.csv file (downloaded from our dataset)
- Line 16: Replace with actual path to your TCT-dataset/images directory (downloaded from our dataset)

3. Save the code, go to ``Qwen3_5`` and run the code there

``python qwen3_5_9b_tct.py``

This will then create a CSV file that contains the image file paths, the probe types (tsc, tad, btp, and tcap), the questions, and Qwen3.5's responses to these questions.
