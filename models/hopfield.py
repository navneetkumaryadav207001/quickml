import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            self.weights += np.outer(p, p)
        self.weights /= self.size
        np.fill_diagonal(self.weights, 0)

    def sign(self, x):
        return np.where(x >= 0, 1, -1)

    def recall(self, pattern, steps=1000):
        state = pattern.copy()
        for _ in range(steps):
            i = np.random.randint(self.size)
            state[i] = self.sign(np.dot(self.weights[i], state))
        return state

    def visualize(self, pattern, noisy_input, recovered):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        self._plot_pattern(pattern, "Original Pattern", ax=axes[0])
        self._plot_pattern(noisy_input, "Noisy Input", ax=axes[1])
        self._plot_pattern(recovered, "Recovered Output", ax=axes[2])
        plt.tight_layout()
        plt.show()

    def _plot_pattern(self, pattern, title="", ax=None):
        pattern = pattern.reshape((int(np.sqrt(self.size)), -1))
        if ax is None:
            ax = plt.gca()
        ax.imshow(pattern, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
