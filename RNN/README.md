# RNN Implementation

This directory contains implementations of Recurrent Neural Network architectures from research papers, focusing on LSTM (Long Short-Term Memory) networks.

## Current Implementation

### LSTM Network

The `lstm.ipynb` notebook implements LSTM networks with the following approaches:

- A from-scratch implementation of LSTM cells
- A PyTorch built-in LSTM implementation
- Training and evaluation on the TimeMachine dataset for text generation

## Implementation Details

The notebook includes:

- **LSTMScratch**: A custom implementation of LSTM cells with:
  - Input gate
  - Forget gate
  - Output gate
  - Memory cell
- **LSTM**: A wrapper around PyTorch's built-in LSTM implementation
- Training with gradient clipping to prevent exploding gradients
- Text generation capabilities using the trained model

## Model Architecture

The LSTM model features:

- 32 hidden units
- Multiple time steps for sequence processing
- Vocabulary-sized input and output dimensions
- Text prediction functionality

## Usage

Open the `lstm.ipynb` notebook in Jupyter or another compatible environment to:

1. Train both LSTM implementations on the TimeMachine dataset
2. Compare performance between custom and built-in implementations
3. Generate text using the trained model with custom prompts

## Dependencies

- PyTorch
- d2l library (version 1.0.3)
- Standard Python data science libraries

## References

The implementation follows the LSTM architecture as described in the original paper:

- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory"

## Future Work

- Implementation of other RNN variants (GRU, Bidirectional RNN)
- Application to other sequence tasks (translation, sentiment analysis)
