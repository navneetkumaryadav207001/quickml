import numpy as np

class Linear:
    def __init__(self, in_features, out_features, lr=0.1):
        self.W = np.random.randn(in_features, out_features) * 0.01  # (in, out)
        self.b = np.zeros((1, out_features))                         # (1, out)
        self.lr = lr
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X                        # Save input for backward
        return X @ self.W + self.b       # (n, out)

    def backward(self, dY):
        # dY is gradient from next layer (n, out)
        self.dW = self.X.T @ dY / self.X.shape[0]   # (in, out)
        self.db = np.mean(dY, axis=0, keepdims=True)  # (1, out)
        dX = dY @ self.W.T                           # (n, in)
        return dX

    def update(self):
        self.W -= self.lr * self.dW
        self.b -= self.lr * self.db

class Sigmoid:
    def forward(self, X):
        self.out = 1 / (1 + np.exp(-X))
        return self.out

    def backward(self, dY):
        return dY * self.out * (1 - self.out)  # element-wise grad

class ReLU:
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

    def backward(self, dY):
        return dY * self.mask

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dLoss):
        for layer in reversed(self.layers):
            dLoss = layer.backward(dLoss)

    def update(self):
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update()

def bce_loss(pred, y):
    eps = 1e-9
    return -np.mean(y * np.log(pred + eps) + (1 - y) * np.log(1 - pred + eps))

def bce_grad(pred, y):
    eps = 1e-9
    return (pred - y) / (pred * (1 - pred) + eps)
