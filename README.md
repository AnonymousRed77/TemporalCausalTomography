# Temporal Causal Tomography

This repository is the official implementation of Temporal Causal Tomography for Vision-Language Models. 

## Getting Started

Download our Temporal Causal Tomography (TCT) dataset [here](https://www.kaggle.com/datasets/anonymousred/temporal-causal-tomography).

We used Conda for this research. Follow [this guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) if you do not have Conda installed.

Determine which VLM you want to experiment with. Go to that VLM's directory and follow the steps in the README file there to set up the environment and run the scripts. All directories contain these:
- A README file
- A yml file
- An inference script

You can also experiment with another VLM that is not included here. Make sure that you create a script that returns a CSV output file containing the following fields:
- ``file_name``: the image file directory
- ``type``: the probe type (tsc, tad, btp, or tcap)
- ``prompt``: the question of the four probes
- ``response``: the model's answer to the question (make sure to have it explain its MCQ choice)

## Evaluation

Use the evaluation scripts in the Evaluation directory. Evaluation proceeds in two empirical tiers. Tier 1 performs exact-match scoring on probe outputs and computes metrics from probe choices alone, with no LLM-judge dependence. Tier 2 supplements Tier 1 with LLM-judge analyses of textual explanations on TAD items. 

We used GPT-4o, Claude 4.6 Sonnet, Gemini 2.5 Pro, and Grok 4.20 as judges to evaluate the TAD responses of the models. More information can be found in the README file of this directory. Warning: This requires **paid subscriptions** to use the APIs.

## Acknowledgment

Our research used these wonderful resources:
- [Claude 4.6 Sonnet](https://www.anthropic.com/news/claude-sonnet-4-6)
- [Gemini 2.5 Pro](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro)
- [GPT-4o](https://openai.com/index/hello-gpt-4o/)
- [Grok 4.20](https://www.mindstudio.ai/models/grok-4-20)
- [InternVL3.5](https://internvl.github.io/blog/2025-08-26-InternVL-3.5/)
- [Kaggle](https://www.kaggle.com/)
- [LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT)
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [Phi-4 Reasoning Vision](https://www.microsoft.com/en-us/research/blog/phi-4-reasoning-vision-and-the-lessons-of-training-a-multimodal-reasoning-model/)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3.5](https://qwen.ai/blog?id=qwen3.5)
