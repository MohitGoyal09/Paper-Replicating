# Paper-Replicating

> **A professional, hands-on repository for replicating foundational machine learning papers from scratch using PyTorch.**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)
![License](https://img.shields.io/badge/License-Educational%20Use-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Module Details](#module-details)
- [Installation & Setup](#installation--setup)
- [Quick Start Guide](#quick-start-guide)
- [Implementation Philosophy](#implementation-philosophy)
- [Project Roadmap](#project-roadmap)
- [Contributing Guidelines](#contributing-guidelines)
- [FAQ & Troubleshooting](#faq--troubleshooting)
- [Citation](#citation)
- [License & Disclaimer](#license--disclaimer)
- [References](#references)

## 🎯 Overview

This repository provides **implementation-first learning** of influential machine learning architectures by replicating foundational research papers. Each module includes:

- **Complete PyTorch implementations** of architectures from original papers
- **From-scratch implementations** to deeply understand core concepts
- **Training and evaluation pipelines** with reproducible results
- **Clear documentation** of design choices, challenges, and practical insights
- **Jupyter notebooks and scripts** for both learning and experimentation

The goal is to bridge the gap between theoretical understanding of research papers and practical implementation skills in modern machine learning frameworks.

## ⭐ Key Features

- 🔬 **Paper-driven Implementation**: Each module is based on a specific research paper with proper citations
- 🎓 **Educational Focus**: Designed for learners to understand architectures from first principles
- 🏗️ **Modular Architecture**: Each module is self-contained with clear, well-documented code
- 🔄 **Multiple Implementation Approaches**: From-scratch implementations alongside framework-optimized versions
- 📊 **Reproducible Results**: Clear metrics tracking and validation methodologies
- 📚 **Comprehensive Documentation**: Detailed READMEs, architecture diagrams, and training guides in each module
- ⚡ **Production-Ready Features**: Advanced training techniques, checkpointing, and inference pipelines
- 🐍 **Pure Python/PyTorch**: No custom CUDA kernels or complex dependencies (except where necessary)

## 📁 Repository Structure

```
Paper-Replicating/
├── AutoEncoders/          # Autoencoder architectures and experiments
│   ├── AutoEncoders.ipynb # Vanilla autoencoder implementation
│   └── README.md
├── CNN/                   # Convolutional Neural Networks from classic papers
│   ├── LeNet.ipynb        # LeNet-5 (LeCun et al., 1998)
│   ├── AlexNet.ipynb      # AlexNet (Krizhevsky et al., 2012)
│   └── README.md
├── RNN/                   # Recurrent Neural Networks and LSTM
│   ├── lstm.ipynb         # LSTM from-scratch and PyTorch implementations
│   └── README.md
├── LLAMA/                 # Large Language Model implementation
│   ├── model.py           # LLaMA architecture
│   ├── train.py           # Basic training script
│   ├── train_advanced.py  # Advanced training with scheduling and amp
│   ├── inference.py       # Text generation and inference
│   ├── data_utils.py      # Data processing utilities
│   ├── config.py          # Configuration management
│   ├── requirements.txt   # Python dependencies
│   └── README.md
└── README.md              # This file
```

## 📚 Module Details

### 1. AutoEncoders (`AutoEncoders/`)

**Paper Reference**: Foundational Autoencoder Principles

**Notebook**: `AutoEncoders.ipynb`

Implements a vanilla autoencoder with:
- Multi-layer encoder-decoder architecture with bottleneck representation
- Training on MNIST dataset for unsupervised learning
- Latent space visualization using PCA and t-SNE
- Reconstruction quality evaluation metrics

**Key Features**:
- Encoder: 1024 → 128 → 64 → 32 → bottleneck dimensions
- ReLU hidden layers, Sigmoid output layer
- MSE loss function
- Interactive visualizations and reconstruction examples

**Future Work**: Variational Autoencoders (VAE), Denoising Autoencoders, Sparse Autoencoders

### 2. Convolutional Neural Networks (`CNN/`)

**Implementations**:

#### LeNet-5 (LeCun et al., 1998)
**Paper**: "Gradient-Based Learning Applied to Document Recognition"

- 5-layer architecture with 2 convolutional layers and 3 fully connected layers
- Average pooling for dimensionality reduction
- Optimized for MNIST digit recognition (32×32 input)
- Implements ReLU activations and proper weight initialization

#### AlexNet (Krizhevsky et al., 2012)
**Paper**: "ImageNet Classification with Deep Convolutional Neural Networks"

- 8-layer deep architecture (5 conv + 3 fully connected)
- Local Response Normalization for improved generalization
- Max pooling with stride for hierarchical feature extraction
- Dropout regularization to prevent overfitting
- Designed for 224×224 ImageNet classification

**Common Features**:
- Layer-by-layer implementations matching original papers
- Proper architectural details and hyperparameters
- Training and evaluation pipelines

**Future Work**: VGG, GoogLeNet/Inception, ResNet, DenseNet

### 3. Recurrent Neural Networks (`RNN/`)

**Paper Reference**: Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"

**Notebook**: `lstm.ipynb`

Implements LSTM networks with:
- **From-scratch LSTM**: Complete custom implementation of LSTM cells with input, forget, and output gates
- **PyTorch LSTM**: Official implementation for performance comparison
- **Text Generation**: Trained on TimeMachine dataset with custom prompt generation
- **Gradient Clipping**: Prevention of exploding gradients during training

**Model Architecture**:
- 32 hidden units (configurable)
- Vocabulary-sized input and output dimensions
- Multi-step sequence processing for temporal dependencies

**Future Work**: GRU, Bidirectional RNN, Attention mechanisms, Sequence-to-sequence models

### 4. Large Language Models (`LLAMA/`)

**Paper Reference**: Touvron et al. (2023) - "LLaMA: Open and Efficient Foundation Language Models"

**Complete Production-Ready Implementation**:

- **Model Architecture**:
  - RMSNorm (Root Mean Square Layer Normalization)
  - RoPE (Rotary Position Embeddings) for position-aware attention
  - Multi-Head Attention with Grouped Query Attention (GQA)
  - SwiGLU activation function in feed-forward networks
  - Pre-normalization (pre-norm) transformer blocks

- **Training Systems**:
  - `train.py`: Basic training with simple checkpointing
  - `train_advanced.py`: Advanced training with:
    - Learning rate scheduling (warmup + cosine decay)
    - Gradient clipping for training stability
    - Mixed precision training (AMP) for GPU efficiency
    - Weights & Biases integration for experiment tracking
    - Early stopping and best model checkpointing

- **Inference & Generation**:
  - Text generation with configurable sampling
  - Prompt-based generation pipeline
  - Support for multiple model sizes

- **Configuration System**:
  - Predefined sizes: small (2M params), medium (15M params), large (100M params)
  - Fully customizable parameters for experiments

**Supported Model Sizes**:

| Size   | Dimension | Layers | Heads | Parameters |
|--------|-----------|--------|-------|------------|
| Small  | 256       | 4      | 4     | ~2M        |
| Medium | 512       | 8      | 8     | ~15M       |
| Large  | 1024      | 16     | 16    | ~100M      |

## 🚀 Installation & Setup

### System Requirements

- **Python**: 3.9 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional but recommended (CUDA 11.8+ for NVIDIA GPUs)
- **OS**: Linux, macOS, or Windows

### Step 1: Clone the Repository

```bash
# Using HTTPS
git clone https://github.com/MohitGoyal09/Paper-Replicating.git
cd Paper-Replicating

# Using SSH
git clone git@github.com:MohitGoyal09/Paper-Replicating.git
cd Paper-Replicating
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Core Dependencies

```bash
# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Or install PyTorch with CUDA support (recommended for GPU)
# Visit https://pytorch.org/get-started/locally/ for your specific configuration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install Repository Dependencies

```bash
# For Jupyter notebook modules (AutoEncoders, CNN, RNN)
pip install jupyter jupyterlab matplotlib numpy scikit-learn scipy seaborn

# For LLAMA module (optional, required only if using LLAMA module)
cd LLAMA
pip install -r requirements.txt
cd ..
```

### Step 5: Verify Installation

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test Jupyter (for notebook modules)
python -m jupyter --version
```

## 🎯 Quick Start Guide

### Option 1: Exploring Notebook-Based Modules (AutoEncoders, CNN, RNN)

```bash
# Start Jupyter
jupyter notebook

# Navigate to the module directory and open the desired notebook:
# - AutoEncoders/AutoEncoders.ipynb
# - CNN/LeNet.ipynb or CNN/AlexNet.ipynb
# - RNN/lstm.ipynb
```

Then:
1. Read the markdown cells for context and theory
2. Run cells sequentially to see implementations and results
3. Modify hyperparameters and re-run to experiment

### Option 2: Training LLAMA Model

```bash
cd LLAMA

# Basic training (uses TinyShakespeare dataset by default)
python train.py

# Advanced training with mixed precision and learning rate scheduling
python train_advanced.py --config medium --use_amp --max_iters 5000

# Generate text using trained model
python inference.py --prompt "To be or not to be"
```

### Option 3: Custom Configuration

```bash
cd LLAMA

# Train with custom parameters
python train_advanced.py \
  --config small \
  --batch_size 32 \
  --learning_rate 0.001 \
  --max_iters 1000 \
  --use_amp
```

## 🔬 Implementation Philosophy

Our approach to replicating research papers follows these principles:

1. **Paper-First Learning**: Study the original research paper thoroughly before implementation
2. **From-Scratch Understanding**: Implement core components from scratch first, then optionally use framework optimizations
3. **Faithful Reproduction**: Maintain architectural fidelity to original papers while adapting for modern PyTorch
4. **Clear Documentation**: Every implementation includes:
   - Architecture diagrams
   - Detailed code comments explaining key concepts
   - Comparison with original paper specifications
   - Empirical validation and metrics

5. **Modular Design**: Each component can be understood independently
6. **Reproducibility**: Fixed random seeds, documented hyperparameters, and clear training procedures
7. **Educational Value**: Prioritize clarity and learning over performance optimizations

## 📈 Expected Results & Metrics

Each module includes validation metrics:

- **AutoEncoders**: Reconstruction MSE, latent space quality, visualization metrics
- **CNN**: Classification accuracy on MNIST/ImageNet, per-layer feature visualizations
- **RNN/LSTM**: Perplexity on text generation, sequence prediction accuracy
- **LLAMA**: Loss curves, text generation coherence, training stability metrics

See individual module READMEs for detailed performance benchmarks.

## 🛣️ Project Roadmap

### Short-term (Q1-Q2 2024)
- [ ] Add comprehensive test suites for all modules
- [ ] Improve benchmark reporting with standardized metrics
- [ ] Add performance comparison tables across implementations

### Medium-term (Q3-Q4 2024)
- [ ] Implement Transformer architectures from papers (Attention is All You Need)
- [ ] Add GAN implementations (DCGANs, StyleGAN concepts)
- [ ] Implement Vision Transformers (ViT)

### Long-term (2025+)
- [ ] Diffusion model implementations
- [ ] Multimodal models (CLIP-like architectures)
- [ ] Continuous integration with automated testing
- [ ] Performance benchmarking against official implementations

## 👥 Contributing Guidelines

We welcome contributions from the community! Please follow these guidelines:

### Before Contributing
1. Check existing issues and PRs to avoid duplicates
2. Ensure your contribution aligns with the repository's educational mission
3. Read the implementation philosophy section above

### How to Contribute

1. **Fork** the repository on GitHub
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Follow the existing code style (PEP 8 for Python)
   - Add comments explaining complex logic
   - Include docstrings for functions and classes
   - Update relevant README files

4. **Test your changes**:
   - Verify the code runs without errors
   - Test edge cases and different inputs
   - Document any new dependencies

5. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: [clear description of change]"
   ```

6. **Push and create a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Contribution Types

We appreciate contributions in these areas:

- **New paper implementations**: CNNs, Vision Transformers, Diffusion models, etc.
- **Bug fixes**: Report with clear reproduction steps
- **Documentation improvements**: Better explanations, tutorials, visualizations
- **Test additions**: Unit tests and integration tests
- **Performance optimizations**: While maintaining code clarity
- **Educational materials**: Blogs, videos, tutorials using these implementations

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use meaningful variable names
- Add type hints where applicable
- Keep functions small and focused (single responsibility principle)
- Use comments for "why", not "what"

## ❓ FAQ & Troubleshooting

### Installation Issues

**Q: I get "ModuleNotFoundError: No module named 'torch'"**
A: PyTorch is not installed. Run:
```bash
pip install torch torchvision torchaudio
```

**Q: CUDA is available but PyTorch isn't using it**
A: Verify CUDA installation and reinstall PyTorch for your CUDA version:
```bash
python -c "import torch; print(torch.cuda.is_available())"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Jupyter Notebook Issues

**Q: Jupyter notebook won't start**
A: Ensure Jupyter is installed:
```bash
pip install jupyter jupyterlab
python -m jupyter notebook
```

**Q: Kernel keeps dying when running cells**
A: Increase memory limits or reduce batch size:
```bash
# In notebook, restart kernel and reduce batch_size parameter
```

### LLAMA Training Issues

**Q: "CUDA out of memory" error during training**
A: Reduce batch size or model dimension:
```bash
python train_advanced.py --config small --batch_size 8
```

**Q: Training loss is unstable or diverging**
A: Use gradient clipping and lower learning rate:
```bash
python train_advanced.py --config medium --learning_rate 0.0005 --gradient_clip 1.0
```

**Q: Very slow training on CPU**
A: Consider using GPU or reducing model size:
```bash
python train_advanced.py --config small --batch_size 8
```

### General Debugging

- **Check Python version**: `python --version` (should be 3.9+)
- **Verify all dependencies**: `pip list`
- **Check PyTorch version**: `python -c "import torch; print(torch.__version__)"`
- **Review module README**: Each module has troubleshooting in its README.md

## 📖 Citation

If you use this repository in your research or educational materials, please cite it as:

```bibtex
@misc{paper_replicating,
  title={Paper-Replicating: Implementation-First Learning of ML Papers},
  author={Goyal, Mohit},
  year={2024},
  publisher={GitHub},
  howpublished{\url{https://github.com/MohitGoyal09/Paper-Replicating}}
}
```

**Also cite the original papers** for each module you use:

- **AutoEncoders**: Original autoencoder papers by Hinton & Salakhutdinov
- **LeNet**: LeCun, Y., et al. (1998). "Gradient-Based Learning Applied to Document Recognition"
- **AlexNet**: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks"
- **LSTM**: Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory"
- **LLaMA**: Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models"

## 📝 License & Disclaimer

### License
This repository is provided for **educational and research purposes only**. While the code and implementations are original, they are based on published research papers. Please respect the intellectual property and licensing of the original papers.

### Disclaimer
- This is an **educational project**, not an official implementation
- Code is provided "as-is" without warranty
- For **production use**, refer to official implementations and licensing terms
- Always **cite original papers** when using implementations from this repository
- Ensure you have the right to use datasets (MNIST, ImageNet, etc.) in your context

### Original Paper Licenses
- Respect the licensing terms of the original research papers
- When citing papers, follow their official citation guidelines
- For commercial applications, verify licensing with original authors/institutions

## 🔗 References & Resources

### Foundational Papers

1. **Hinton & Salakhutdinov (2006)** - "Reducing the Dimensionality of Data with Neural Networks"
2. **LeCun et al. (1998)** - "Gradient-Based Learning Applied to Document Recognition"
3. **Krizhevsky et al. (2012)** - "ImageNet Classification with Deep Convolutional Neural Networks"
4. **Hochreiter & Schmidhuber (1997)** - "Long Short-Term Memory"
5. **Touvron et al. (2023)** - "LLaMA: Open and Efficient Foundation Language Models"

### Learning Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [Deep Learning Book](https://www.deeplearningbook.org/) - Goodfellow, Bengio, Courville
- [CS231n: CNNs](http://cs231n.stanford.edu/) - Stanford University
- [CS224n: NLP](http://web.stanford.edu/class/cs224n/) - Stanford University
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper

### Related Projects

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Hub](https://www.tensorflow.org/hub)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## 📞 Support & Community

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/MohitGoyal09/Paper-Replicating/issues)
- **Discussions**: Join community discussions for questions and ideas
- **Contribute**: See [Contributing Guidelines](#contributing-guidelines)

---

**Last Updated**: 2024
**Maintainer**: [Mohit Goyal](https://github.com/MohitGoyal09)

If you find this repository helpful, please star ⭐ it to show your support!
