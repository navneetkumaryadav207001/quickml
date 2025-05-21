import numpy as np

class Perceptron:
    def __init__(self, X, y, lr=0.1):
        self.X = X                          # shape: (n_samples, n_features)
        self.y = y                          # shape: (n_samples,)
        self.w = np.zeros(X.shape[1])       # weights for each feature
        self.c = 1                         # scalar bias
        self.grad = np.zeros(X.shape[1] +1) # [grad_w..., grad_c]
        self.lr = lr                         # learning rate

    def sigmoid(self, n):
        return 1 / (1 + np.exp(-n))

    def forward(self, X=None):
        if X is None:
            X = self.X
        z = np.dot(X, self.w) + self.c
        return self.sigmoid(z)

    def backward(self):
        y_hat = self.forward()
        dL_dz = y_hat - self.y              # shape: (n_samples,)
        grad_w = np.dot(self.X.T, dL_dz) / self.X.shape[0]  # shape: (n_features,)
        grad_c = np.mean(dL_dz)                              # scalar
        self.grad[:-1] = grad_w
        self.grad[-1] = grad_c
        return self.grad

    def update(self):
        self.w -= self.lr * self.grad[:-1]
        self.c -= self.lr * self.grad[-1]
