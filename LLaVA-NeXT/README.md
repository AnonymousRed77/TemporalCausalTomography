# LLaVA-NeXT

Follow the following steps to set up the enviromment and run the inference for LLaVA-NeXT. 

## Minimum configuration

We used 1 NVIDIA A40 GPU with 48GB to run and fine-tune this model.

## Getting Started

1. Clone the GitHub repo

``git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git``

2. Go to the ``LLaVA-NeXT`` directory
   
``cd LLaVA-NeXT``

3. Download the ``llava_next.yml`` file and move it to the ``LLaVA-NeXT`` directory

4. Create and activate the environment

``conda env create -f llava_next.yml``

``conda activate llava_next``

## Inference

1. Download the ``llavanext_13b_tct.py`` script and move it to ``LLaVA-NeXT``

2. Open ``llavanext_13b_tct.py`` and modify the code under 2 TODOs in this script

- Line 11: Replace with actual path to your metadata.csv file (downloaded from our dataset)
- Line 14: Replace with actual path to your TCT-dataset/images directory (downloaded from our dataset)

3. Save the code, go to ``LLaVA-NeXT`` and run the code there

``python llavanext_13b_tct.py``

This will then create a CSV file that contains the image file paths, the types of questions (tsc, tad, btp, tcap), the questions, and LLaVA-NeXT's responses to these questions.
