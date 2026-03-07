"""
MNIST Training Example

Demonstrates training a neural network on handwritten digit classification.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))  # Add parent dir to path to import nn

from nn import NeuralNetwork, Dense, BatchNorm, ReLU, Dropout, Softmax, load_mnist_data

# For quick testing, use n_train=5000, n_test=1000
# For full training, use n_train=None, n_test=None
X_train, y_train, X_test, y_test = load_mnist_data(
    n_train=None,
    n_test=None,
    one_hot=True,
    normalize=True
)

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
print("\nTraining on MNIST...")
model.train(
    X_train, y_train,
    epochs=20,
    lr=0.001,
    batch_size=64,
    loss='categorical',
    lr_decay='step',
    step_size=10,
    decay_rate=0.5
)

test_acc = model.evaluate(X_test, y_test, loss='categorical')
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")

predictions = model.predict(X_test[:10])
actual = y_test[:10].argmax(axis=1) if len(y_test.shape) > 1 else y_test[:10]

print(f"\nFirst 10 predictions: {predictions.tolist()}")
print(f"Actual labels:        {actual.tolist()}")