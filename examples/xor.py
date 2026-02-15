"""
XOR Training Example

Demonstrates training a simple neural network on the XOR problem.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))  # Add parent dir to path to import nn

from nn import NeuralNetwork, Dense, BatchNorm, ReLU, Dropout, Softmax, load_xor_data

X_train, y_train = load_xor_data(n_samples=100, one_hot=True)
X_test, y_test = load_xor_data(n_samples=4, one_hot=True)

model = NeuralNetwork([
    Dense(2, 16, use_adam=True),
    BatchNorm(16),
    ReLU(),
    Dropout(0.3),
    Dense(16, 2, use_adam=True),
    Softmax()
])

# Train
print("Training on XOR...")
model.train(
    X_train, y_train,
    epochs=2000,
    lr=0.01,
    batch_size=16,
    loss='categorical',
    verbose=False
)

# Evaluate
acc = model.evaluate(X_test, y_test, loss='categorical')
preds = model.predict(X_test)

print(f"\nTest Accuracy: {acc * 100:.1f}%")
print(f"Predictions: {preds.tolist()}")
print(f"Expected:    [0, 1, 1, 0]")