import numpy as np
import random

class simpleLogisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w = np.zeros(X.shape[1])
        self.b = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_prob(self, x):
        return self.sigmoid(np.dot(x, self.w) + self.b)

    def predict(self, x):
        return self.predict_prob(x) >= 0.5

    def train(self, epochs, lr):
        for i in range(epochs):
            random_n = random.randint(0, len(self.X) - 1)
            x_i = self.X[random_n]
            y_i = self.y[random_n]

            pred = self.predict_prob(x_i)
            error = pred - y_i

            # Gradient descent step
            self.w -= lr * error * x_i
            self.b -= lr * error

            if i % (epochs // 100) == 0 and i != 0:
                print("-", end="", flush=True)
