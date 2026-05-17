# Paper-Replicating

A professional, hands-on repository for replicating foundational machine learning papers in PyTorch.

## Overview

This project focuses on understanding research papers through implementation-first learning.  
Each module aims to reproduce key architectural ideas, build training/evaluation workflows, and document practical findings.

## Objectives

- Recreate influential ML architectures from original papers
- Implement models and training loops from scratch using PyTorch
- Track meaningful metrics for validation
- Document design choices, challenges, and results clearly

## Repository Structure

- `AutoEncoders/` — Autoencoder implementations and experiments
- `CNN/` — CNN paper replications (e.g., LeNet, AlexNet)
- `RNN/` — RNN/LSTM implementations and sequence modeling experiments
- `LLAMA/` — LLaMA-style transformer implementation with training and inference scripts

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch
- Jupyter Notebook (for notebook-based modules)

### Clone the Repository

```bash
git clone https://github.com/MohitGoyal09/Paper-Replicating.git
cd Paper-Replicating
```

### Run Notebook-Based Modules

Open notebooks under `AutoEncoders/`, `CNN/`, and `RNN/` using Jupyter.

### Run the LLaMA Module

```bash
cd LLAMA
pip install -r requirements.txt
python train.py
```

## Implementation Philosophy

Each replication generally follows:

1. Paper study and architecture breakdown
2. PyTorch implementation of core model components
3. Training and evaluation setup
4. Metric-based validation
5. Documentation of insights and limitations

## Roadmap

- Add more replication tracks (e.g., Transformers, GANs, diffusion models)
- Improve benchmark reporting and reproducibility
- Standardize experiment tracking across modules

## Contributing

Contributions are welcome. Please open an issue or pull request with a clear description of the proposed change.

## Disclaimer

This repository is intended for educational and research purposes.  
Please refer to original papers and official licenses for production or commercial usage constraints.
