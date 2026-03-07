import numpy as np

class BatchNorm:
  def __init__(self, num_features):
    self.gamma = np.ones((1, num_features))
    self.beta  = np.zeros((1, num_features))
    self.running_mean = np.zeros((1, num_features))
    self.running_var  = np.ones((1, num_features))
    self.X_norm = np.zeros((1, num_features))
    self.X_var = np.ones((1, num_features))
    self.X_mean = np.zeros((1, num_features))
    self.dgamma = np.zeros_like(self.gamma)
    self.dbeta  = np.zeros_like(self.beta)

  def forward(self, X, training=True):
    eps = 1e-8
    if training:
      m = np.mean(X, axis=0)
      v = np.var(X, axis=0)

      self.running_mean = 0.9 * self.running_mean + 0.1 * m
      self.running_var = 0.9 * self.running_var + 0.1 * v
    else:
      m = self.running_mean
      v = self.running_var

    X_norm = (X - m) / np.sqrt(v + eps)
    X_out = self.gamma * X_norm + self.beta

    self.X_norm = X_norm
    self.X_var  = v
    self.X_mean = m

    return X_out

  def backward(self, dout):
    eps = 1e-8
    n = dout.shape[0]
    X_norm = self.X_norm
    var = self.X_var
    mean = self.X_mean
    X = X_norm * np.sqrt(var + eps) + mean

    dgamma = np.sum(dout * X_norm, axis=0, keepdims=True)
    dbeta  = np.sum(dout, axis=0, keepdims=True)

    self.dgamma = dgamma
    self.dbeta  = dbeta

    dX_norm = dout * self.gamma
    dvar = np.sum(dX_norm * (X - mean) * -0.5 * (var + eps) ** -1.5, axis=0, keepdims=True)
    dmean = np.sum(dX_norm * -1 / np.sqrt(var + eps), axis=0, keepdims=True)
    dX = dX_norm / np.sqrt(var + eps) + dvar * 2 * (X - mean) / n + dmean / n

    return dX

  def update(self, lr):
    self.gamma -= lr * self.dgamma
    self.beta -= lr * self.dbeta



class Dropout:
  def __init__(self, dropout_rate=0.5):
    self.dropout = dropout_rate
    self.dropout_mask = []

  def forward(self, X, training=True):
    if not training or self.dropout == 0:
      return X

    mask = (np.random.rand(*X.shape) > self.dropout).astype(float)
    X_out = X * mask
    self.dropout_mask = mask
    return X_out / (1 - self.dropout)

  def backward(self, dout):
    return dout * self.dropout_mask / (1 - self.dropout)

  def update(self, lr):
      pass