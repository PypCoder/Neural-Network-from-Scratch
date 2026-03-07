import numpy as np
from .layers.activations import Sigmoid, Softmax

class NeuralNetwork:
  """
  Layer-based neural network.
  
  Usage:
      model = NeuralNetwork([
          Dense(784, 128),
          BatchNorm(128),
          ReLU(),
          Dropout(0.5),
          Dense(128, 10),
          Softmax()
      ])
  """
  def __init__(self, layers):
    self.layers = layers

  def forward(self, X, training=True):
    """Pass input through all layers."""
    for layer in self.layers:
        X = layer.forward(X, training=training)
    return X

  def backward(self, dout):
    """Backpropagate gradient through all layers in reverse."""
    for layer in reversed(self.layers):
        dout = layer.backward(dout)

  def update(self, lr):
    """Update all learnable parameters."""
    for layer in self.layers:
      layer.update(lr)
  
  def lr_decay(self, lr, epoch, mode='none', decay_rate=0.5, step_size=10):
    if mode == 'none':
        return lr
    elif mode == 'step':
        if epoch % step_size == 0 and epoch > 0:
            return lr * decay_rate
        return lr                               
    elif mode == 'exponential':
        return lr * (decay_rate ** epoch)
    elif mode == '1/t':
        return lr / (1 + decay_rate * epoch)

  def train(self, X, y, epochs=20, lr=0.001, batch_size=64, 
            lr_decay='none', decay_rate=0.5, step_size=10, 
            loss='categorical', verbose=True):
    
    """ Train the network. """
    
    n = X.shape[0]
    
    for epoch in range(epochs):
      indices = np.random.permutation(n)
      X, y = X[indices], y[indices]

      epoch_loss = 0
      num_batches = 0

      current_lr = self.lr_decay(lr, epoch=epoch,mode=lr_decay, 
                                 decay_rate=decay_rate,step_size=step_size)

      for start in range(0, n, batch_size):
          X_batch = X[start:start + batch_size]
          y_batch = y[start:start + batch_size]

          y_pred = self.forward(X_batch, training=True)

          loss_val = self._compute_loss(y_batch, y_pred, loss)
          epoch_loss += loss_val

          dout = y_pred - y_batch
          self.backward(dout)

          self.update(lr=current_lr)

          num_batches += 1

      if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
          avg_loss = epoch_loss / num_batches
          print(f"Epoch {epoch+1:>4}/{epochs}  |  Loss: {avg_loss:.4f}  |  LR: {current_lr}")

  def predict(self, X):
    """Make predictions (returns class indices for classification)."""
    y_pred = self.forward(X, training=False)
    
    if isinstance(self.layers[-1], Softmax):
        return np.argmax(y_pred, axis=1)
    elif isinstance(self.layers[-1], Sigmoid):
        return np.round(y_pred)
    else:
        return y_pred

  def evaluate(self, X, y, loss='categorical'):
    """Evaluate accuracy or loss."""
    y_pred = self.forward(X, training=False)
    
    if loss == 'categorical':
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
    elif loss == 'binary':
        return np.mean(np.round(y_pred) == y)
    elif loss == 'mse':
        return self._compute_loss(y, y_pred, loss='mse')

  def _compute_loss(self, y, y_pred, loss):
    """Compute loss value."""
    eps = 1e-8
    if loss == 'categorical':
        y_pred = np.clip(y_pred, eps, 1)
        return -np.mean(np.sum(y * np.log(y_pred), axis=1))
    elif loss == 'binary':
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    elif loss == 'mse':
        return np.mean((y - y_pred) ** 2)



def to_onehot(y, num_classes):
  one_hot = np.zeros((y.size, num_classes))
  one_hot[np.arange(y.size), y.flatten()] = 1
  return one_hot