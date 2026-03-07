import numpy as np

class ReLU:
  def forward(self, X, training=True):
    self.X = X
    return np.maximum(0, X)

  def backward(self, dout):
    return dout * (self.X > 0)

  def update(self, lr):
    pass

class Sigmoid:
  def forward(self, X, training=True):
    self.out = 1 / (1 + np.exp(-X))
    return self.out

  def backward(self, dout):
    return dout * self.out * (1 - self.out)

  def update(self, lr):
    pass

class Softmax:
  def forward(self, X, training=True):
    X = X - np.max(X, axis=1, keepdims=True)
    exp_X = np.exp(X)
    self.out = exp_X / np.sum(exp_X, axis=1, keepdims=True)
    return self.out

  def backward(self, dout):
    return dout

  def update(self, lr):
    pass