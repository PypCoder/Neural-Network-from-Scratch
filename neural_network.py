import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, output='softmax', batch_norm = True):
        """
        layer_sizes : list of ints  e.g. [784, 128, 64, 10]
        output      : 'softmax'  → multiclass  (use with 'categorical')
                      'sigmoid'  → binary       (use with 'binary')
                      'linear'   → regression   (use with 'mse')
        """
        np.random.seed(42)
        self.output = output
        self.batch_norm = batch_norm
        self.weights = []
        self.biases = []
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []
        self.t = 0
        self.gamma = [np.ones((1, size))  for size in layer_sizes[1:-1]]
        self.beta  = [np.zeros((1, size)) for size in layer_sizes[1:-1]]
        self.running_mean = [np.zeros((1, size)) for size in layer_sizes[1:-1]]
        self.running_var  = [np.ones((1, size))  for size in layer_sizes[1:-1]]
        self.Z_norm = [np.zeros((1, size)) for size in layer_sizes[1:-1]]
        self.Z_var = [np.ones((1, size)) for size in layer_sizes[1:-1]]
        self.Z_mean = [np.zeros((1, size)) for size in layer_sizes[1:-1]]
        self.dgamma = [np.zeros_like(g) for g in self.gamma]
        self.dbeta  = [np.zeros_like(b) for b in self.beta]

        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

            self.m_w.append(np.zeros_like(W))
            self.v_w.append(np.zeros_like(W))

            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))

    # ------------------------------------------------------------------ #
    #  Activations                                                         #
    # ------------------------------------------------------------------ #
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)   # numerical stability
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # x here is already sigmoid output A, not Z
        return x * (1 - x)

    # ------------------------------------------------------------------ #
    #  Loss                                                                #
    # ------------------------------------------------------------------ #
    def compute_loss(self, y, y_pred, loss='categorical'):
        """
        loss: 'categorical' → softmax output,  one-hot y
              'binary'      → sigmoid output,  binary y  (0 or 1)
              'mse'         → linear output,   continuous y
        """
        eps = 1e-8
        if loss == 'categorical':
            y_pred = np.clip(y_pred, eps, 1)
            return -np.mean(np.sum(y * np.log(y_pred), axis=1))

        elif loss == 'binary':
            y_pred = np.clip(y_pred, eps, 1 - eps)
            return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        elif loss == 'mse':
            return np.mean((y - y_pred) ** 2)

        else:
            raise ValueError(f"Unknown loss: '{loss}'. Use 'categorical', 'binary', or 'mse'.")

    def batchnorm_forward(self, Z, layer_idx, training=True):
      eps = 1e-8
      if training:
        m = np.mean(Z, axis=0)
        v = np.var(Z, axis=0)

        self.running_mean[layer_idx] = 0.9 * self.running_mean[layer_idx] + 0.1 * m
        self.running_var[layer_idx] = 0.9 * self.running_var[layer_idx] + 0.1 * v
      else:
        m = self.running_mean[layer_idx]
        v = self.running_var[layer_idx]

      Z_norm = (Z - m) / np.sqrt(v + eps)
      Z_out = self.gamma[layer_idx] * Z_norm + self.beta[layer_idx]

      self.Z_norm[layer_idx] = Z_norm
      self.Z_var[layer_idx]  = v
      self.Z_mean[layer_idx] = m

      return Z_out

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(self, X, training=True):
        """
        Hidden layers  → always ReLU
        Output layer   → determined by self.output ('softmax', 'sigmoid', 'linear')
        """
        self.A = [X]
        self.Z = []

        for i in range(len(self.weights)):
            z = self.A[i] @ self.weights[i] + self.biases[i]

            if i == len(self.weights) - 1:          # output layer
                if self.output == 'softmax':
                    a = self.softmax(z)
                elif self.output == 'sigmoid':
                    a = self.sigmoid(z)
                else:                               # linear
                    a = z
            else:                                   # hidden layers
              if self.batch_norm:
                z = self.batchnorm_forward(z, layer_idx=i, training=training)
              a = self.relu(z)

            self.Z.append(z)
            self.A.append(a)

        return self.A[-1]

    # ------------------------------------------------------------------ #
    #  Backward                                                            #
    # ------------------------------------------------------------------ #

    def batchnorm_backward(self, dZ_bn, layer_idx):
      eps   = 1e-8
      n     = dZ_bn.shape[0]
      Z_norm = self.Z_norm[layer_idx]
      var    = self.Z_var[layer_idx]
      mean   = self.Z_mean[layer_idx]
      Z      = Z_norm * np.sqrt(var + eps) + mean  # recover original Z

      # gradients for gamma and beta
      dgamma = np.sum(dZ_bn * Z_norm, axis=0, keepdims=True)
      dbeta  = np.sum(dZ_bn, axis=0, keepdims=True)

      # store for update later
      self.dgamma[layer_idx] = dgamma
      self.dbeta[layer_idx]  = dbeta

      # gradient to pass back through normalization
      dZ_norm = dZ_bn * self.gamma[layer_idx]
      dvar    = np.sum(dZ_norm * (Z - mean) * -0.5 * (var + eps) ** -1.5, axis=0, keepdims=True)
      dmean   = np.sum(dZ_norm * -1 / np.sqrt(var + eps), axis=0, keepdims=True)
      dZ      = dZ_norm / np.sqrt(var + eps) + dvar * 2 * (Z - mean) / n + dmean / n

      return dZ

    def backward(self, y, optimizer='Adam', lr=0.001):
        """
        Output layer gradient depends on output activation + loss pairing:
          softmax  + categorical → dZ = A - y   (simplified combined derivative)
          sigmoid  + binary      → dZ = A - y   (also simplifies cleanly)
          linear   + mse         → dZ = A - y   (MSE derivative, no activation to chain)

        All three simplify to the same thing — A - y.
        Hidden layers always use ReLU derivative.
        """
        n  = y.shape[0]
        L  = len(self.weights)
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8

        # output layer — all three pairings simplify to A - y
        dA = self.A[-1] - y

        for i in reversed(range(L)):
            if i == L - 1:
                dZ = dA                                     # output layer
            else:
                dZ = dA * self.relu_derivative(self.Z[i])  # hidden layers
                if self.batch_norm:
                  dZ = self.batchnorm_backward(dZ, layer_idx=i)

            dW = self.A[i].T @ dZ / n
            db = np.sum(dZ, axis=0, keepdims=True) / n
            dA = dZ @ self.weights[i].T                     # pass gradient back
            if optimizer == 'Adam':
              self.m_w[i] = b1 * self.m_w[i] + (1 - b1) * dW
              self.v_w[i] = b2 * self.v_w[i] + (1 - b2) * (dW ** 2)
              m_w_hat = self.m_w[i] / (1 - b1 ** self.t)
              v_w_hat = self.v_w[i] / (1 - b2 ** self.t)
              self.m_b[i] = b1 * self.m_b[i] + (1 - b1) * db
              self.v_b[i] = b2 * self.v_b[i] + (1 - b2) * (db ** 2)
              m_b_hat = self.m_b[i] / (1 - b1 ** self.t)
              v_b_hat = self.v_b[i] / (1 - b2 ** self.t)
              self.weights[i] -= lr * (m_w_hat / (np.sqrt(v_w_hat) + eps))
              self.biases[i]  -= lr * (m_b_hat / (np.sqrt(v_b_hat) + eps))
            else:
              self.weights[i] -= lr * dW
              self.biases[i]  -= lr * db
        # after the loop — update gamma and beta same as weights
        if self.batch_norm:  
          for i in range(len(self.gamma)):
              self.gamma[i] -= lr * self.dgamma[i]
              self.beta[i]  -= lr * self.dbeta[i]

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def to_onehot(self, y, num_classes):
        one_hot = np.zeros((y.size, num_classes))
        one_hot[np.arange(y.size), y.flatten()] = 1
        return one_hot

    def sample_data(self, mode='binary'):
        """
        XOR dataset.
        mode: 'binary'      → y shape (4,1)  for sigmoid + binary loss
              'categorical' → y shape (4,2)  for softmax + categorical loss
        """
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        y = np.array([[0],[1],[1],[0]])
        if mode == 'binary':
            return X, y
        else:
            return X, self.to_onehot(y, num_classes=2)

    def lr_decay(self, lr, epoch, mode='none', decay_rate=0.5, step_size=10):
      if mode == 'none':
          return lr
      elif mode == 'step':
          if epoch % step_size == 0 and epoch > 0:   # ← boundary check lives here
              return lr * decay_rate
          return lr                                   # ← unchanged otherwise
      elif mode == 'exponential':
          return lr * (decay_rate ** epoch)
      elif mode == '1/t':
          return lr / (1 + decay_rate * epoch)

    # ------------------------------------------------------------------ #
    #  Train                                                               #
    # ------------------------------------------------------------------ #
    def train(self, X, y, epochs=20, lr=0.001, batch_size=64, 
          loss='categorical', optimizer='Adam', verbose=True,
          lr_decay='none', decay_rate=0.5, step_size=10):
        self.t = 0
        n = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            X, y = X[indices], y[indices]

            epoch_loss  = 0
            num_batches = 0

            current_lr = self.lr_decay(lr, epoch=epoch, 
                                       mode=lr_decay, decay_rate=decay_rate, 
                                       step_size=step_size)

            for start in range(0, n, batch_size):
              X_batch = X[start:start + batch_size]
              y_batch = y[start:start + batch_size]

              y_pred = self.forward(X_batch, training=True)
              epoch_loss += self.compute_loss(y_batch, y_pred, loss=loss)
              self.backward(y_batch, lr=current_lr, optimizer=optimizer)
              num_batches += 1

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch+1:>4}/{epochs}  |  Loss: {avg_loss:.4f}  |  LR: {current_lr}")

    # ------------------------------------------------------------------ #
    #  Evaluate & Predict                                                  #
    # ------------------------------------------------------------------ #
    def evaluate(self, X, y, loss='categorical'):
        """Returns accuracy for classification, MSE for regression."""
        y_pred = self.forward(X, training=False)
        if loss == 'categorical':
            return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        elif loss == 'binary':
            return np.mean(np.round(y_pred) == y)
        elif loss == 'mse':
            return self.compute_loss(y, y_pred, loss='mse')

    def predict(self, X):
        y_pred = self.forward(X, training=False)
        if self.output == 'softmax':
            return np.argmax(y_pred, axis=1)
        elif self.output == 'sigmoid':
            return np.round(y_pred)
        else:
            return y_pred