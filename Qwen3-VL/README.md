# Qwen3-VL

Follow the following steps to set up the enviromment and run the inference for Qwen3-VL (both 8B and 30B versions). 

## Minimum configuration

We used 1 NVIDIA A40 GPU with 48GB to run this model.

## Getting Started

1. Create a dirctory and name it ``Qwen3-VL``

2. Go to the ``Qwen3-VL`` directory
   
``cd Qwen3-VL``

3. Download the ``qwen3vl8b.yml``/``qwen3vl30b.yml`` file and move it to the ``Qwen3-VL`` directory

4. Create and activate the environment

``conda env create -f qwen3vl8b.yml``/``conda env create -f qwen3vl30b.yml``

``conda activate qwen3vl8b``/``conda activate qwen3vl30b``

## Inference

1. Download the ``qwen3vl_8b_tct.py``/``qwen3vl_30b_tct.py`` script and move it to ``Qwen3-VL``

2. Open ``qwen3vl_8b_tct.py``/``qwen3vl_30b_tct.py`` and modify the code under 2 TODOs in this script

- Line 11: Replace with actual path to your metadata.csv file (downloaded from our dataset)
- Line 14: Replace with actual path to your TCT-dataset/images directory (downloaded from our dataset)

3. Save the code, go to ``Qwen3-VL`` and run the code there

``python qwen3vl_8b_tct.py``/``python qwen3vl_30b_tct.py``

This will then create a CSV file that contains the image file paths, the probe types (tsc, tad, btp, and tcap), the questions, and Qwen3-VL's responses to these questions.
