# AutoEncoders Implementation

This directory contains implementations of various autoencoder architectures from research papers.

## Current Implementation

### Vanilla Autoencoder

The `AutoEncoders.ipynb` notebook implements a basic non-linear autoencoder with the following features:

- Encoder-decoder architecture with multiple fully connected layers
- Bottleneck representation for dimensionality reduction
- Training on the MNIST dataset
- Visualization of latent space representations

## Architecture Details

The implemented autoencoder has:

- **Encoder**: A series of fully connected layers (1024→128→64→32→bottleneck_size)
- **Decoder**: A series of fully connected layers (bottleneck_size→32→64→128→1024)
- **Activation Functions**: ReLU for hidden layers, Sigmoid for output layer
- **Loss Function**: Mean Squared Error (MSE)

## Training Process

The notebook includes:

- Data preprocessing and loading using PyTorch DataLoaders
- Training loop with evaluation at regular intervals
- Visualization of latent space using PCA and t-SNE
- Reconstruction quality evaluation

## Usage

Open the `AutoEncoders.ipynb` notebook in Jupyter or another compatible environment to:

1. Train the autoencoder model
2. Visualize the latent space representations
3. Generate reconstructions from the trained model

## References

The implementation is based on the fundamental principles of autoencoders as described in various research papers.

## Future Work

- Implementation of Variational Autoencoders (VAE)
- Denoising Autoencoders
- Sparse Autoencoders
