# Neural Network from Scratch

A fully functional feedforward neural network built using only NumPy — no PyTorch, no TensorFlow, no shortcuts.

Built iteratively as a learning project, implementing every component by hand including forward propagation, backpropagation, Adam optimizer, Batch Normalization, and Learning Rate Decay.

**Achieves 97.93% accuracy on MNIST.**

---

## What's implemented

- Dynamic architecture — define any depth/width via a layer list e.g. `[784, 128, 64, 10]`
- He weight initialization
- Activations: ReLU (hidden layers), Sigmoid, Softmax, Linear (output)
- Loss functions: Categorical Cross-Entropy, Binary Cross-Entropy, MSE
- Optimizers: Adam, SGD
- Mini-batch gradient descent
- Batch Normalization (toggleable via `batch_norm=True/False`)
- Learning Rate Decay — Step, Exponential, and 1/t schedules
- Supports binary classification, multiclass classification, and regression

---

## Usage

```python
from neural_network import NeuralNetwork

# Multiclass — MNIST with Batch Norm + LR Decay
nn = NeuralNetwork([784, 128, 64, 10], output='softmax', batch_norm=True)
nn.train(X_train, y_train, epochs=20, lr=0.001, batch_size=64,
         loss='categorical', optimizer='Adam',
         lr_decay='step', decay_rate=0.5, step_size=10)
print(nn.evaluate(X_test, y_test, loss='categorical'))

# Binary — XOR
nn = NeuralNetwork([2, 4, 1], output='sigmoid', batch_norm=False)
nn.train(X, y, epochs=2000, lr=0.01, batch_size=4, loss='binary', optimizer='Adam')
print(nn.predict(X))
```

---

## Results

| Dataset | Architecture       | Optimizer | Batch Norm | LR Decay | Epochs | Accuracy |
|---------|--------------------|-----------|------------|----------|--------|----------|
| XOR     | [2, 8, 2]          | Adam      | ✅         | —        | 3000   | 100%     |
| MNIST   | [784, 128, 64, 10] | Adam      | ✅         | Step     | 20     | 97.93%   |

---

## Install

```bash
pip install numpy scikit-learn
```

---

## Structure

```
neural-network-from-scratch/
├── neural_network.py        # full implementation
├── nn_from_scratch.ipynb    # development notebook with experiments
├── requirements.txt
└── README.md
```

---

## Roadmap

This is an active learning project. Planned additions:

- [x] Learning rate decay
- [x] Batch Normalization
- [ ] Dropout regularization
- [ ] Modular layer-based architecture (v2)
- [ ] Paper-style documentation with full derivations

---

> Built from scratch to actually understand what's happening — not just call `.fit()`.

<p align="center">
  <a href="https://github.com/PypCoder" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-PypCoder-181717?style=for-the-badge&logo=github&logoColor=white" alt="PypCoder GitHub"/>
  </a>
</p>