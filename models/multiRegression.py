import numpy as np
class multiRegression:
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")
        self.X = X
        self.y = y
        self.m = np.zeros(X.shape[1]) 
        self.b = 0 

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.m) + self.b

    def train(self, epochs: int, alpha: float,log:bool = False) -> None:
        if log:
            print("Multivariate Model Training ->")
        N = self.X.shape[0]  
        for i in range(epochs):
            predictions = self.predict(self.X)
            errors = self.y - predictions
            SME_m = -np.dot(self.X.T, errors)  
            SME_b = -np.sum(errors)
            self.m -= alpha * (1 / N) * SME_m
            self.b -= alpha * (1 / N) * SME_b
            if i % (epochs // 100) == 0 and i != 0 and log:
                print("-", end="", flush=True)
        if log:
            print("Done") 