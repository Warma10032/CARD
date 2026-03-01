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
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐│
│  │   Agents    │  │   Tools     │  │   Dynamic Information   ││
│  │ - CodeWriter│  │ - Search    │  │ - LLM Profiles          ││
│  │ - MathSolver│  │ - Executor  │  │ - Tool Capabilities     ││
│  │ - Analyze   │  │ - RAG       │  │ - Knowledge Sources    ││
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘│
│         │                │                      │              │
│         └────────────────┼──────────────────────┘              │
│                          ▼                                     │
│              ┌───────────────────────┐                         │
│              │   Graph Neural Network │                         │
│              │   (GCN + Feature Fusion)│                        │
│              └───────────┬───────────┘                         │
│                          ▼                                     │
│              ┌───────────────────────┐                         │
│              │   Dynamic Graph       │                         │
│              │   - Spatial Edges    │                         │
│              │   - Temporal Edges   │                         │
│              └───────────────────────┘                         │
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

## Quick Start

```python
from CARD.graph.graph import Graph

# Create a graph with multiple agents
graph = Graph(
    domain="math",
    llm_name="gpt-4",
    agent_names=["MathSolver", "AnalyzeAgent"],
    decision_method="FinalRefer",
    optimized_spatial=True,
    optimized_temporal=True,
    num_agents=5
)

# Run the graph
result = graph.run({"task": "Solve this math problem..."})
```

## Supported Agents

| Agent                | Description                              |
| -------------------- | ---------------------------------------- |
| `CodeWriting`      | Generates and debugs code                |
| `MathSolver`       | Solves mathematical problems             |
| `AnalyzeAgent`     | Analyzes and provides insights           |
| `FinalDecision`    | Makes final decisions from agent outputs |
| `AdversarialAgent` | Tests robustness with adversarial inputs |

## Supported Tools

### Search Tools

- Google Search
- DuckDuckGo
- Baidu
- Wikipedia
- arXiv

### Code Execution

- Python Executor
- Code validation

### Information Retrieval

- RAG (Dense, Sparse, Hybrid modes)
- Multiple knowledge sources (PDF, Web, Database)

## Experiments

Run benchmarks on standard datasets:

### MMLU (Massive Multitask Language Understanding)

```bash
python experiments/run_mmlu.py \
    --phase train \
    --llm_name gpt-4o-mini \
    --mode FullConnected \
    --num_iterations 10
```

### HumanEval (Code Generation)

```bash
python experiments/run_humaneval.py \
    --phase train \
    --llm_name DeepSeek-V3 \
    --mode FullConnected
```

### GSM8K (Math Reasoning)

```bash
python experiments/run_gsm8k.py \
    --phase train \
    --llm_name gpt-4o-mini
```

## Graph Modes

CARD supports various graph topologies:

- **DirectAnswer**: Single agent direct response
- **FullConnected**: All agents connected to each other
- **Chain**: Sequential agent pipeline
- **Debate**: Agents discuss and vote
- **Layered**: Hierarchical agent structure
- **Star**: Central agent coordinates others
- **Random**: Random connections (for comparison)

## Configuration

Configure agents and node layouts in JSON files:

```json
{
  "group_1": [
    {"role": "Math Expert", "llm_name": "gpt-4"},
    {"role": "Code Expert", "llm_name": "gpt-4"}
  ]
}
```

## Key Components

### Graph Module (`CARD/graph/`)

- `graph.py`: Core graph structure with GNN integration
- `node.py`: Individual agent nodes

### Agents (`CARD/agents/`)

- `agent_registry.py`: Agent registration system
- Multiple specialized agents for different tasks

### Tools (`CARD/tools/`)

- `search/`: Web search integration
- `coding/`: Code execution
- `reader/`: Document parsing

### Dynamic Information (`CARD/dynamic/`)

- `llm_information.py`: LLM capability profiles
- `search.py`: Search engine information
- `rag.py`: RAG configuration details

### LLM Integration (`CARD/llm/`)

- `gpt_chat.py`: OpenAI API integration
- `together_chat.py`: Together AI integration
- `llm_registry.py`: Multi-LLM support

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
