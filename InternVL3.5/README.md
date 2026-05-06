# InternVL3.5

Follow the following steps to set up the enviromment and run the inference for InternVL3.5 (both 8B and 30B versions). 

## Minimum configuration

We used 1 NVIDIA A40 GPU with 48GB to run this model.

## Getting Started

1. Create a dirctory and name it ``InternVL3_5``

2. Go to the ``InternVL3_5`` directory
   
``cd InternVL3_5``

3. Download the ``internvl.yml`` file and move it to the ``InternVL3_5`` directory

4. Create and activate the environment

``conda env create -f internvl.yml``

``conda activate internvl``

## Inference

1. Download the ``internvl3_5_8b_tct.py``/``internvl3_5_30b_tct.py`` script and move it to ``InternVL3_5``

2. Open ``internvl3_5_8b_tct.py``/``internvl3_5_30b_tct.py`` and modify the code under 2 TODOs in this script

- Line 15: Replace with actual path to your metadata.csv file (downloaded from our dataset)
- Line 18: Replace with actual path to your TCT-dataset/images directory (downloaded from our dataset)

3. Save the code, go to ``InternVL3_5`` and run the code there

``python internvl3_5_8b_tct.py``/``python internvl3_5_30b_tct.py``

This will then create a CSV file that contains the image file paths, the probe types (tsc, tad, btp, and tcap), the questions, and InternVL3.5's responses to these questions.
