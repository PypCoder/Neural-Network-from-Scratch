# Neural Network from Scratch

A fully functional feedforward neural network built using only NumPy — no PyTorch, no TensorFlow, no shortcuts.

Built iteratively as a learning project, implementing every component by hand including forward propagation, backpropagation, Adam optimizer, Batch Normalization, Dropout, and Learning Rate Decay. Features a clean, modular layer-based architecture.

**Achieves 97.93% accuracy on MNIST.**

---

## What's implemented

- **Modular Layer-Based Architecture**: Compose networks from independent layer objects
- **Layers**: Dense (fully connected), ReLU, Sigmoid, Softmax, Batch Normalization, Dropout
- **Optimizers**: Adam (with momentum + RMSprop), SGD
- **Regularization**: Dropout, Batch Normalization, L2 (via weight decay)
- **Learning Rate Schedules**: Step decay, Exponential decay, 1/t decay
- **Loss Functions**: Categorical Cross-Entropy, Binary Cross-Entropy, MSE
- **Initialization**: He initialization for ReLU networks
- **Training Features**: Mini-batch gradient descent, training/test mode switching
- **Supports**: Binary classification, multiclass classification, regression

---

## Quick Start

### Installation
```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### Usage

```python
from nn import NeuralNetwork, Dense, BatchNorm, ReLU, Dropout, Softmax
from nn.data.mnist import load_data

# Load MNIST
X_train, y_train, X_test, y_test = load_data(n_train=5000, n_test=1000)

# Build model with layer composition
model = NeuralNetwork([
    Dense(784, 128, use_adam=True),
    BatchNorm(128),
    ReLU(),
    Dropout(0.5),
    Dense(128, 64, use_adam=True),
    BatchNorm(64),
    ReLU(),
    Dropout(0.5),
    Dense(64, 10, use_adam=True),
    Softmax()
])

# Train
model.train(
    X_train, y_train,
    epochs=20,
    lr=0.001,
    batch_size=64,
    loss='categorical',
    lr_decay='step',
    step_size=10
)

# Evaluate
acc = model.evaluate(X_test, y_test, loss='categorical')
print(f"Test Accuracy: {acc * 100:.2f}%")
```

---

## Examples

Run the included examples:
```bash
# XOR problem (quick sanity check)
uv run examples/xor.py

# MNIST digit classification (full training)
uv run examples/mnist.py
```

Or using standard Python:
```bash
python examples/xor.py
python examples/mnist.py
```

Both examples use flexible data loaders that let you control dataset size:

```python
# XOR with 100 noisy samples
from nn.data.xor import load_data
X, y = load_data(n_samples=100, one_hot=True)

# MNIST with 5000 training samples
from nn.data.mnist import load_data
X_train, y_train, X_test, y_test = load_data(n_train=5000, n_test=1000)
```

---

## Results

| Dataset | Architecture       | Optimizer | Batch Norm | Dropout | LR Decay | Epochs | Accuracy |
|---------|--------------------|-----------|------------|---------|----------|--------|----------|
| XOR     | [2, 16, 2]         | Adam      | ✅         | 30%     | —        | 2000   | 100%     |
| MNIST   | [784, 128, 64, 10] | Adam      | ✅         | 50%     | Step     | 20     | 97.93%   |

---

## Project Structure

```
neural-network-from-scratch/
├── nn/                          # Core package
│   ├── __init__.py              # Clean imports
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── core.py              # Dense layer
│   │   ├── activations.py       # ReLU, Sigmoid, Softmax
│   │   └── regularization.py    # BatchNorm, Dropout
│   ├── datasets/
│   │   ├── xor.py               # XOR data loader
│   │   └── mnist.py             # MNIST data loader
│   └── network.py               # NeuralNetwork class
├── examples/
│   ├── xor.py                   # Runnable XOR example
│   └── mnist.py                 # Runnable MNIST example
├── notebooks/
│   └── nn_from_scratch.ipynb    # Development notebook
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Architecture Highlights

### Modular Layer Design

Every component is a self-contained layer with three methods:

```python
class Layer:
    def forward(self, X, training=True):  # Compute output
    def backward(self, dout):             # Compute gradients
    def update(self, lr):                 # Update parameters
```

This makes the network composable and extensible:

```python
# Define network as a list of layers
model = NeuralNetwork([
    Dense(input_size, hidden_size),
    BatchNorm(hidden_size),
    ReLU(),
    Dropout(0.5),
    Dense(hidden_size, output_size),
    Softmax()
])
```

### Training Loop

Clean separation of concerns:

```python
y_pred = model.forward(X, training=True)  # Forward pass
loss = compute_loss(y, y_pred)            # Compute loss
model.backward(y_pred - y)                # Backpropagation
model.update(lr)                          # Update weights
```

---

## Roadmap

- [x] Learning rate decay
- [x] Batch Normalization
- [x] Dropout regularization
- [x] Modular layer-based architecture

---

## Why This Exists

> Built from scratch to actually understand what's happening, not just call `.fit()`.

This project is a learning journey through neural network fundamentals. Every line of code was written to understand the math, not to match framework performance. If you want production code, use PyTorch. If you want to understand backpropagation, start here.

---

<p align="center">
  <a href="https://github.com/PypCoder" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-PypCoder-181717?style=for-the-badge&logo=github&logoColor=white" alt="PypCoder GitHub"/>
  </a>
</p>
