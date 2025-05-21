import numpy as np

class linearRegression:
    def __init__(self,X:np.ndarray,y:np.ndarray) -> None:
        if not isinstance(X,np.ndarray) and isinstance(y,np.ndarray):
            raise ValueError("X and y must be numpy arrays")
        self.X = X
        self.y = y
        self.m = 0
        self.b = 0

    def predict(self,x:float) -> float:
        return self.m*x+self.b
    
    def train(self,epochs:int,alpha:float,log:bool = False)->None:
        if log:
            print("Linear Model Training ->")
        for i in range(epochs):
            predictions = self.m * self.X + self.b
            errors = self.y - predictions 
            SME_m = -np.sum(self.X * errors)
            SME_b = -np.sum(errors)
            self.m -= alpha * 1/self.X.size * SME_m
            self.b -= alpha * 1/self.X.size * SME_b

            # progress bar
            if i % (epochs // 100) == 0 and i != 0:
                    if log:
                        print("-", end="", flush=True) 
        if log:
            print("Done")