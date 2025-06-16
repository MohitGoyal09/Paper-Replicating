# CNN Implementation

This directory contains implementations of Convolutional Neural Network architectures from foundational research papers.

## Current Implementations

### LeNet-5
The `LeNet.ipynb` notebook implements the LeNet-5 architecture as described in the paper:
- LeCun, Y., et al. (1998). "Gradient-Based Learning Applied to Document Recognition"

Features:
- 5-layer architecture (2 convolutional layers, 3 fully connected layers)
- Average pooling layers
- ReLU activation functions
- Input size of 32x32 for MNIST digit recognition

### AlexNet
The `AlexNet.ipynb` notebook implements the AlexNet architecture as described in the paper:
- Krizhevsky, A., et al. (2012). "ImageNet Classification with Deep Convolutional Neural Networks"

Features:
- 8-layer architecture (5 convolutional layers, 3 fully connected layers)
- Local Response Normalization
- Max Pooling layers
- ReLU activation functions
- Dropout regularization
- Input size of 224x224 for image classification

## Implementation Details

Both implementations include:
- PyTorch model definitions with proper initialization
- Layer-by-layer architecture following the original papers
- Forward pass implementations

## Usage

Open the respective notebooks in Jupyter or another compatible environment to:
1. Examine the model architectures
2. Train the models on appropriate datasets
3. Evaluate model performance

## References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-based learning applied to document recognition"
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet classification with deep convolutional neural networks"

## Future Work

- Implementation of other influential CNN architectures:
  - VGG
  - GoogLeNet/Inception
  - ResNet
  - DenseNet
- Training and evaluation on standard datasets
- Performance comparison between architectures
- Visualization of learned features 