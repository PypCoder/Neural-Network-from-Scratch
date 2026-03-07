import numpy as np

class Dense:
  def __init__(self, input_size, output_size, use_adam=True):
    self.W = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
    self.b = np.zeros((1, output_size))
    self.use_adam = use_adam
    if use_adam:
        self.t = 0
        self.m_w = np.zeros_like(self.W)
        self.v_w = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)

  def forward(self, X, training=True):
    self.X = X
    return X @ self.W + self.b

  def backward(self, dout):
    n = self.X.shape[0]
    self.dW = self.X.T @ dout / n
    self.db = np.sum(dout, axis=0, keepdims=True) / n
    return dout @ self.W.T

  def update(self, lr):
    if self.use_adam:
      self.t += 1
      b1, b2, eps = 0.9, 0.999, 1e-8
      self.m_w = b1 * self.m_w + (1 - b1) * self.dW
      self.v_w = b2 * self.v_w + (1 - b2) * (self.dW ** 2)
      m_w_hat = self.m_w / (1 - b1 ** self.t)
      v_w_hat = self.v_w / (1 - b2 ** self.t)
      self.m_b = b1 * self.m_b + (1 - b1) * self.db
      self.v_b = b2 * self.v_b + (1 - b2) * (self.db ** 2)
      m_b_hat = self.m_b / (1 - b1 ** self.t)
      v_b_hat = self.v_b / (1 - b2 ** self.t)
      self.W -= lr * (m_w_hat / (np.sqrt(v_w_hat) + eps))
      self.b -= lr * (m_b_hat / (np.sqrt(v_b_hat) + eps))
    else:
      self.W -= lr * self.dW
      self.b -= lr * self.db