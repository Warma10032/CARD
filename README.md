# CARD: Towards Conditional Design of Multi-agent Topological Structures

<p align="center">
  <a href="https://github.com/Warma10032/CARD">
    <img src="https://img.shields.io/badge/GitHub-Repository-blue?style=flat-square" alt="GitHub">
  </a>
  <a href="https://openreview.net/pdf?id=JgvJdICc6P">
    <img src="https://img.shields.io/badge/Paper-PDF-green?style=flat-square" alt="Paper">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.3+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

CARD (Conditional Design of Multi-agent Topological Structures) is a novel framework that leverages Large Language Models (LLMs) to build collaborative agent systems using dynamic graph structures. It integrates Graph Neural Networks (GNNs) with multi-agent systems to enable intelligent agent collaboration and reasoning.

**Authors**: Tongtong Wu, Yanming Li, Ziye Tang, Chen Jiang, Linhao Luo, Guilin Qi, Shirui Pan, Gholamreza Haffari

## Features

- **Dynamic Graph-Based Agent Collaboration**: Build agent networks with learnable spatial and temporal connections
- **GNN-Enhanced Reasoning**: Use Graph Neural Networks to optimize agent collaboration patterns
- **Flexible Agent System**: Support for various agent types including code writing, mathematical reasoning, analysis, and more
- **External Tool Integration**: Built-in tools for web search, code execution, RAG, and more
- **Multiple LLM Support**: Compatible with GPT-4, Claude, DeepSeek, Llama, and other models
- **Benchmark Evaluation**: Ready-to-use experiments on MMLU, HumanEval, and GSM8K datasets

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CARD Framework                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Agents    │  │   Tools     │  │   Dynamic Information   │  │
│  │ - CodeWriter│  │ - Search    │  │ - LLM Profiles          │  │
│  │ - MathSolver│  │ - Executor  │  │ - Tool Capabilities     │  │
│  │ - Analyze   │  │ - RAG       │  │ - Knowledge Sources     │  │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘  │
│         │                │                      │               │
│         └────────────────┼──────────────────────┘               │
│                          ▼                                      │
│              ┌───────────────────────────┐                      │
│              │   Graph Neural Network    │                      │
│              │   (GCN + Feature Fusion)  │                      │
│              └───────────┬───────────────┘                      │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │   Dynamic Graph       │                          │
│              │   - Spatial Edges     │                          │
│              │   - Temporal Edges    │                          │
│              └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Warma10032/CARD.git
cd CARD

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (copy from template.env)
cp template.env .env
# Edit .env with your API keys
```

## Experiments

Run benchmarks on standard datasets:

### MMLU

#### Train

```bash
python experiments/run_mmlu.py \
    --phase train \
    --eval_group "cycle" \
    --mode FullConnected \
    --num_iterations 10 \
    --agent_nums 5 \
    --batch_size 8 \
    --optimized_spatial
```

#### Eval

```
python experiments/run_mmlu.py \
    --phase eval \
    --eval_group "model_group_1" \
    --mode FullConnected \
    --num_iterations 10 \
    --agent_nums 5 \
    --batch_size 8 \
    --optimized_spatial
```

## Configuration

Configure agents and node layouts in JSON files:

```json
{
  "model_group_1": [
    {"role": "Math Expert", "llm_name": "gpt-4o"},
    {"role": "Code Expert", "llm_name": "gpt-4o"}
  ]
}
```

## Requirements

- Python 3.10+
- PyTorch 2.3+
- Transformers
- PyTorch Geometric
- OpenAI API key (or other LLM providers)

## Citation

If you use CARD in your research, please cite:

```bibtex
@inproceedings{card2026,
  title = {CARD: Towards Conditional Design of Multi-agent Topological Structures},
  author = {Tongtong Wu and Yanming Li and Ziye Tang and Chen Jiang and Linhao Luo and Guilin Qi and Shirui Pan and Gholamreza Haffari},
  booktitle = {ICLR},
  year = {2026}
}
```

Or cite the paper directly:

- [Paper PDF](https://openreview.net/pdf?id=JgvJdICc6P)

## Acknowledgments

- Various open-source search and tool providers
- This code refers to [GDesigner](https://github.com/yanweiyue/GDesigner)
