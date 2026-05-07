# Proprietary Models (GPT-4o, Claude 4.6 Sonnet, Gemini 2.5 Pro)

**WARNING**: Running the inference for these models requires paid subscriptions to their APIs.

Follow the following steps to set up the enviromment and run the inference for the three proprietary models. 

## Getting Started

1. Create a dirctory and name it ``Proprietary-Models``

2. Go to the ``Proprietary-Models`` directory
   
``cd Proprietary-Models``

3. Download the ``proprietary_models.yml`` file and move it to the ``Proprietary-Models`` directory

4. Create and activate the environment

``conda env create -f proprietary_models.yml``

``conda activate proprietary_models``

## Inference

### GPT-4o

1. Download the ``gpt4_tct.py`` script and move it to ``Proprietary-Models``

2. Open ``gpt4_tct.py`` and modify the code under 2 TODOs in this script

- Line 29: Replace with actual path to your metadata.csv file (downloaded from our dataset)
- Line 32: Replace with actual path to your TCT-dataset/images directory (downloaded from our dataset)

3. Create an OpenAI API key and set it as an environment variable using the following command

``export OPENAI_API_KEY='YOUR-API-KEY'``

4. Save the code, go to ``Proprietary-Models`` and run the code there

``python gpt4_tct.py``

This will then create a CSV file that contains the image file paths, the probe types (tsc, tad, btp, and tcap), the questions, and GPT-4o's responses to these questions.

### Claude 4.6 Sonnet

1. Download the ``claude_tct.py`` script and move it to ``Proprietary-Models``

2. Open ``claude_tct.py`` and modify the code under 2 TODOs in this script

- Line 32: Replace with actual path to your metadata.csv file (downloaded from our dataset)
- Line 35: Replace with actual path to your TCT-dataset/images directory (downloaded from our dataset)

3. Create an Anthropic API key and set it as an environment variable using the following command

``export ANTHROPIC_API_KEY='YOUR-API-KEY'``

4. Save the code, go to ``Proprietary-Models`` and run the code there

``python claude_tct.py``

This will then create a CSV file that contains the image file paths, the probe types (tsc, tad, btp, and tcap), the questions, and Claude 4.6 Sonnet's responses to these questions.

### Gemini 2.5 Pro

1. Download the ``gemini_tct.py`` script and move it to ``Proprietary-Models``

2. Open ``gemini_tct.py`` and modify the code under 2 TODOs in this script

- Line 15: Replace with actual path to your metadata.csv file (downloaded from our dataset)
- Line 18: Replace with actual path to your TCT-dataset/images directory (downloaded from our dataset)

3. Create an Gemini API key and set it as an environment variable using the following command

``export GEMINI_API_KEY='YOUR-API-KEY'``

4. Save the code, go to ``Proprietary-Models`` and run the code there

``python gemini_tct.py``

This will then create a CSV file that contains the image file paths, the probe types (tsc, tad, btp, and tcap), the questions, and Gemini 2.5 Pro's responses to these questions.
