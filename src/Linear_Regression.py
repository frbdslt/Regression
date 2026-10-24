import numpy as np

class LinearRegression:
    def __init__(self,learning_rate=0.01,iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.cost_history = []


    def fit(self,X,y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X] 
        m,n_plus_one = X_b.shape
        self.w = np.zeros((n_plus_one,1))
        self.cost_history = []

        for _ in range(self.iterations):
            y_pred = np.dot(X_b,self.w)
            error = y_pred - y

            gradient = (1/m) * np.dot(X_b.T, error)
            self.w -= self.learning_rate * gradient

            cost = (1/(2*m)) * np.sum(error ** 2)
            self.cost_history.append(cost)
    def predict(self,X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X] 
        return np.dot(X_b,self.w)        
