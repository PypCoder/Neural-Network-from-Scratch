# Neural Network from Scratch

A fully functional feedforward neural network built using only NumPy — no PyTorch, no TensorFlow, no shortcuts.

Built iteratively as a learning project, implementing every component by hand including forward propagation, backpropagation, and the Adam optimizer.

**Achieves 97.78% accuracy on MNIST.**

---

## What's implemented

- Dynamic architecture — define any depth/width via a layer list e.g. `[784, 128, 64, 10]`
- He weight initialization
- Activations: ReLU (hidden layers), Sigmoid, Softmax, Linear (output)
- Loss functions: Categorical Cross-Entropy, Binary Cross-Entropy, MSE
- Optimizers: Adam, SGD
- Mini-batch gradient descent
- Supports binary classification, multiclass classification, and regression

---

## Usage

```python
from neural_network import NeuralNetwork

# Multiclass — MNIST
nn = NeuralNetwork([784, 128, 64, 10], output='softmax', optimizer='Adam', lr=0.001)
nn.train(X_train, y_train, epochs=20, batch_size=64, loss='categorical')
print(nn.evaluate(X_test, y_test, loss='categorical'))

# Binary — XOR
nn = NeuralNetwork([2, 4, 1], output='sigmoid', optimizer='Adam', lr=0.01)
nn.train(X, y, epochs=2000, batch_size=4, loss='binary')
print(nn.predict(X))
```

---

## Results

| Dataset | Architecture     | Optimizer | Epochs | Accuracy |
|---------|-----------------|-----------|--------|----------|
| XOR     | [2, 4, 2]       | Adam      | 2000   | 100%     |
| MNIST   | [784, 128, 64, 10] | Adam   | 20     | 97.78%   |

---

## Install

```bash
pip install numpy scikit-learn
```

---

## Structure

```
neural-network-from-scratch/
├── neural_network.py     # full implementation
├── nn_from_scratch.ipynb        # development notebook with experiments
├── requirements.txt
└── README.md
```

---

## Roadmap

This is an active learning project. Planned additions:

- [ ] Learning rate decay
- [ ] Batch Normalization
- [ ] Dropout regularization
- [ ] Modular layer-based architecture (v2)
- [ ] Paper-style documentation with full derivations

---

> Built from scratch to actually understand what's happening — not just call ```.fit()```.

<p align="center">
  <a href="https://github.com/PypCoder" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-PypCoder-181717?style=for-the-badge&logo=github&logoColor=white" alt="PypCoder GitHub"/>
  </a>
</p>