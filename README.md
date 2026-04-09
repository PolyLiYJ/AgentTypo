# AgentTypo: Adaptive Typographic Prompt Injection Attacks against Black-box Multimodal Agents

Official code and data of our paper:<br>
**AgentTypo: Adaptive Typographic Prompt Injection Attacks against Black-box Multimodal Agents** <br>
Yanjie Li, Yiming Cao, Dong Wang, Bin Xiao<br>
Hong Kong Polytechnic University <br>

[**[Paper link]**](https://arxiv.org/abs/2510.04257) | [**[Data]**](https://chenwu.io/attack-agent/)

<br>
<div align=center>
    <img src="docs/pipeline.png" align="middle">
</div>
<br>

## Abstract

Multimodal agents built on large vision–language models (LVLMs) are increasingly deployed in open-world settings but remain highly vulnerable to prompt injection, especially through visual inputs. We introduce **AgentTypo**, a black-box red-teaming framework that mounts adaptive typographic prompt injection by embedding optimized text into webpage images. Our **Automatic Typographic Prompt Injection (ATPI)** algorithm maximizes prompt reconstruction by substituting captioners while minimizing human detectability via a stealth loss, with a Tree-structured Parzen Estimator guiding black-box optimization over text placement, size, and color. To further enhance attack strength, we develop **AgentTypo-pro**, a multi-LLM system that iteratively refines injection prompts using evaluation feedback and retrieves successful past examples for continual learning.

### Key Results

| Model | Attack Type | AgentAttack | AgentTypo (Ours) |
|-------|-------------|-------------|------------------|
| GPT-4o | Image-only | 23% | **45%** |
| GPT-4o | Image+Text | - | **68%** |

## Contents

- [AgentTypo](#agentyppo-adaptive-typographic-prompt-injection-attacks-against-black-box-multimodal-agents)
  - [Abstract](#abstract)
    - [Key Results](#key-results)
  - [Contents](#contents)
  - [Installation](#installation)
    - [Install VisualWebArena](#install-visualwebarena)
    - [Install this repository](#install-this-repository)
  - [Additional Setup](#additional-setup)
    - [Setup API Keys](#setup-api-keys)
    - [Setup experiment directory](#setup-experiment-directory)
  - [Usage](#usage)
    - [Run attacks](#run-attacks)
    - [Setup for episode-wise evaluation](#setup-for-episode-wise-evaluation)
    - [Episode-wise evaluation](#episode-wise-evaluation)
    - [Stepwise evaluation](#stepwise-evaluation)
    - [Lifelong Attack](#lifelong-attack)
    - [Computer Use Agent Testing](#computer-use-agent-testing)
    - [Multi-Model Testing](#multi-model-testing)
  - [Known Issues](#known-issues)
  - [Citation](#citation)

## Installation

Our code requires two repositories, including this one. The file structure should look like this:

```plaintext
.
├── agent-attack  # This repository
└── visualwebarena
```

### Install VisualWebArena

> Can skip this step if you only want to run the lightweight [step-wise evaluation](#stepwise-evaluation) (e.g., for early development) or the [attacks](#run-attacks).

VisualWebArena is required if you want to run the episode-wise evaluation that reproduces the results in our paper.
It requires at least 200GB of disk space and docker to run.

The original version of VisualWebArena can be found [here](https://github.com/web-arena-x/visualwebarena), but we [modified it](https://github.com/ChenWu98/visualwebarena) to support perturbation to the trigger images. Clone the modified version and install:

```bash
git clone git@github.com:ChenWu98/visualwebarena.git
cd visualwebarena/
# Install based on the README.md of https://github.com/ChenWu98/visualwebarena
# Make sure that `pytest -x` passes
```

### Install this repository

Clone the repository and install with pip:

```bash
git clone git@github.com:ChenWu98/agent-attack.git
cd agent-attack/
python -m pip install -e .
```

You may need to install PyTorch according to your CUDA version.

## Additional Setup

### Setup API Keys

> [!IMPORTANT]
> Need to set up the corresponding API keys each time before running the code.

Configurate the OpenAI API key.

```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

If using Claude, configurate the Anthropic API key.

```bash
export ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

If using Gemini, first install the [gcloud CLI](https://cloud.google.com/sdk/docs/install).
Setup a Google Cloud project and get the ID at the [Google Cloud console](https://console.cloud.google.com/).
Get the AI Studio API key from the [AI Studio console](https://aistudio.google.com/app/apikey).
Authenticate Google Cloud and configure the AI Studio API key:

```bash
gcloud auth login
gcloud config set project <your-google-cloud-project-id>
export VERTEX_PROJECT=<your-google-cloud-project-id>  # Same as above
export AISTUDIO_API_KEY=<your-aistudio-api-key>
```

### Setup experiment directory

> Only need to do this once.

Copy the raw data files to the experiment data directory:

```bash
scp -r data/ exp_data/
```
The adversarial examples will later be saved to the `exp_data/` directory.


### Set VisualWebareana Environment
```bash
cd ../visualwebarena/classifieds_docker_compose/
sudo docker-compose restart
sudo docker start forum
sudo docker start shopping
```


## Run Attacks

### AgentTypo Base Attack

```bash
# Run typography attack with ATPI algorithm
python scripts/Bayes_Typography.py

# Run with specific model and instruction file
python scripts/Bayes_Typography.py --model qwen-max --instruction_path agent/prompts/jsons/qwen_p_som_cot_id_actree_3s.json

# The ATPI algorithm optimizes:
# - Text placement (position)
# - Text size (font size)
# - Text color (stealth optimization)
# - Stealth loss (minimize human detectability)
```
> **Note:** For Qwen models, use the `qwen_p_som_cot_id_actree_3s.json` instruction file which includes enhanced action format constraints to prevent invalid actions like "inspect".

### Lifelong Attack (AgentTypo-Pro)

The LifelongAttack framework implements continuous adversarial attacks on multimodal agents through an iterative optimization loop.

LifelongAttack uses a multi-component system:

1. **Strategy Library**: A repository of attack strategies that grows over time
2. **Attacker LLM**: Generates injection prompts using strategies and history
3. **Summarizer LLM**: Analyzes successful attacks to extract new strategies
4. **Embedding Retriever**: Retrieves similar past examples and strategies
5. **Scorer**: Evaluates attack effectiveness

```bash
# use gemini model
python scripts/LifelongAttack.py --model gpt-4o --typo_attack_type fixed

# use Gemini model
python scripts/LifelongAttack.py --model gemini-2.5-flash --typo_attack_type fixed

# use claude model
python scripts/LifelongAttack.py --model claude-3-5-sonnet-20241022 --typo_attack_type fixed

# using auto method with bayesian optimization
python scripts/LifelongAttack.py \
    --attack typography_attack \
    --model gpt-4o-mini \
    --typo_attack_type auto \
    --caption_model qwen-vl-max \
    --typo_n_trials 5 \
    --max_tasks 50 \
    --max_iters 10

# Features:
# - Iterative prompt refinement using evaluation feedback
# - Retrieval of successful past examples (RAG-based)
# - Continual learning across attacks
# - Strategy repository for knowledge accumulation
# - Automatic strategy extraction and generalization
```

### Attack Pipeline

The lifelong attack loop works as follows:

1. **Strategy Retrieval**: Retrieve relevant strategies from the library based on agent responses
2. **Prompt Generation**: Attacker LLM generates injection prompt using strategies and history
3. **Injection & Evaluation**: Inject prompt into webpage image and evaluate agent response
4. **Strategy Summarization**: If score improves, summarize new strategy and add to library
5. **Repeat**: Continue until success threshold or max iterations reached

### Initial Strategy Library

The framework starts with these built-in strategies:

| Strategy | Description |
|----------|-------------|
| Instruction Override | Use forceful phrases to override prior instructions |
| Roleplay & Jailbreak | Prompt model to act as unrestricted persona |
| Contextual Pollution | Insert hidden malicious instructions as context |
| Social Engineering | Use deceptive language to trick the agent |
| Payload Splitting | Split attack into multiple reconstructable parts |
| Encoding/Obfuscation | Encode payloads (Base64, URL, charcode) |
| Logic & Architecture Exploits | Exploit workflow steps or planning modules |
| State Confusion & Memory Poisoning | Manipulate agent's state or memory |
| Imitate Normal Tone | Blend malicious instructions into normal content |
| Negate Correct Information | Contradict accurate information with false data |


### Analyze Successful Attacks

Analyze successful attack cases and test on multiple models:

```bash
# Analyze attacks from lifelong results directory
python analyze_successful_attacks.py --lifelong_dir cache/LifeLong-results_20250902192358_gpt-4o-mini_typography_attack_0

# Test on specific models
python analyze_successful_attacks.py --test_models gpt-5,gpt-4o,gpt-4o-mini

# List all supported models
python analyze_successful_attacks.py --list_models
```

### GPT-5 Model Support

The following GPT-5 models are supported:

| Model | Vision | Temperature |
|-------|--------|-------------|
| gpt-5-chat-latest | ✓ | 1.0 (auto) |
| gpt-5.1 | ✓ | 1.0 (auto) |

> **Note:** GPT-5 models automatically use temperature=1.0 as required by the API.

### Environment Setup for GPT-5

Before running GPT-5 tests, ensure the environment is configured:

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"
# Or use openaikey.txt file
```

## Computer Use Agent Testing

Computer Use agents are multimodal models that can perform tasks by outputting coordinates (bounding boxes) on images, without relying on auxiliary text like accessibility trees.

### Image-Only Attack Testing

Test attacks on Computer-Use models using only image inputs:

```bash
# Test typography attacks on multiple models
python test_imageonly_attacks.py --test_models "gpt-4o,claude-sonnet-4-6" --attack typography

# Test CLIP-based attacks
python test_imageonly_attacks.py --test_models "gemini-2.5-flash" --attack clipattack

# Test on specific models with custom attack types
python test_imageonly_attacks.py --test_models "gpt-4o" --attack typography --typo_attack_type auto
```

### Supported Models for Image-Only Testing

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-5, gpt-5-chat-latest |
| Google | gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash, gemini-2.5-flash |
| Anthropic | claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-sonnet-4-6, claude-opus-4-6 |
| Qwen | qwen-max, qwen-plus |

### Claude Computer Use Evaluation

Evaluate Claude Computer Use agents with coordinate-based output:

```bash
# Evaluate Claude Sonnet 4.5
python eval_claude_computer_use.py --model claude-sonnet-4-5-20250929

# Evaluate Claude Opus 4.6
python eval_claude_computer_use.py --model claude-opus-4-6

# Evaluate Claude Haiku 4.5
python eval_claude_computer_use.py --model claude-haiku-4-5-20251001
```

### GPT Computer Use Evaluation

Evaluate GPT models with coordinate-based output:

```bash
# Evaluate GPT-5 models
python eval_gpt_computer_use.py --models "gpt-5,gpt-5.1"

# Evaluate GPT-4o
python eval_gpt_computer_use.py --models "gpt-4o,gpt-4o-mini"
```

### How Computer Use Evaluation Works

1. **Screenshot Capture**: Takes screenshots of webpage states
2. **Coordinate Prediction**: Model outputs bounding box coordinates for actions
3. **SoM Compatibility**: Converts coordinates to Set-of-Mark (SoM) format
4. **Action Execution**: Executes clicks, types, and other actions
5. **Task Success**: Evaluates if the task was completed successfully

---

## Multi-Model Testing

Test attacks on GPT-5, GPT-4o, and Gemini models:

```bash
# Test on GPT-5.1 (use --mode chat for GPT models)
python analyze_successful_attacks.py --test_models gpt-5.1 --mode chat

# Test on gpt-5-chat-latest (use --mode chat for GPT models)
python analyze_successful_attacks.py --test_models gpt-5-chat-latest --mode chat

# Test on Gemini-2.5-Flash (use --mode completion for Gemini)
python analyze_successful_attacks.py --test_models gemini-2.5-flash --mode completion

# Test on multiple models at once
python analyze_successful_attacks.py --test_models gpt-5.1,gpt-5-chat-latest,gemini-2.5-flash
```

### Model-Specific Configuration

| Model | Provider | Mode | Instruction Path |
|-------|----------|------|------------------|
| gpt-5.1 | openai | chat | agent/prompts/jsons/p_gpt5_multimodal_cot_id_actree_3s.json |
| gpt-5-chat-latest | openai | chat | agent/prompts/jsons/p_gpt5_multimodal_cot_id_actree_3s.json |
| gemini-3-flash-preview | google | completion | agent/prompts/jsons/qwen_p_som_cot_id_actree_3s.json |
| gemini-3.1-flash-lite-preview | google | completion | agent/prompts/jsons/qwen_p_som_cot_id_actree_3s.json |


### Important Notes

- **GPT models** (`gpt-5.1`, `gpt-5-chat-latest`): Use `--mode chat`
- **Gemini models** (`gemini-2.5-flash`): Use `--mode completion`
- The provider is automatically detected based on the model name
- GPT-5 models automatically use temperature=1.0 as required by the API


## Known Issues

See the ``FIXME`` comments in the code for some hard-coded hacks we used to work around slight differences in the environment.

## Citation

If you find this code useful, please consider citing our paper:

```bibtex
@article{li2025agentyppo,
  title={AgentTypo: Adaptive Typographic Prompt Injection Attacks against Black-box Multimodal Agents},
  author={Li, Yanjie and Cao, Yiming and Wang, Dong and Xiao, Bin},
  journal={https://arxiv.org/abs/2510.04257},
  year={2025}
}
```

## Acknowledgments

This project builds upon the foundational work of:

**Dissecting Adversarial Robustness of Multimodal LM Agents**  
Chen Henry Wu, Rishi Shah, Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried, Aditi Raghunathan  
Carnegie Mellon University  
*ICLR 2025* (also *Oral presentation at NeurIPS 2024 Open-World Agents Workshop*)

```bibtex
@article{wu2024agentattack,
  title={Dissecting Adversarial Robustness of Multimodal LM Agents},
  author={Wu, Chen Henry and Shah, Rishi and Koh, Jing Yu and Salakhutdinov, Ruslan and Fried, Daniel and Raghunathan, Aditi},
  journal={arXiv preprint arXiv:2406.12814},
  year={2024}
}
```

We thank the original authors for their open-source contribution to the VisualWebArena benchmark and adversarial attack framework, which made this research possible.

- **Original Paper:** [https://arxiv.org/abs/2406.12814](https://arxiv.org/abs/2406.12814)
- **Original Code:** [https://github.com/ChenWu98/agent-attack](https://github.com/ChenWu98/agent-attack)
- **Website:** [https://chenwu.io/attack-agent/](https://chenwu.io/attack-agent/)
