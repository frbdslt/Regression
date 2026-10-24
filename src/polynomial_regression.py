import numpy as np
from src.Linear_Regression import LinearRegression

class PolynomialRegressionScratch:
    def __init__(self, degree=2, learning_rate=0.05, iterations=2000):
        self.degree = degree
        self.linear_model = LinearRegression(learning_rate=learning_rate, iterations=iterations)
        
    def _generate_polynomial_features(self, X):
        m, n = X.shape
        features = [X]
        
        if self.degree >= 2:
            X_squared = X ** 2
            features.append(X_squared)

            interactions = []
            for i in range(n):
                for j in range(i + 1, n):
                    inter = (X[:, i] * X[:, j]).reshape(-1, 1)
                    interactions.append(inter)
            if interactions:
                features.append(np.hstack(interactions))
                
        return np.hstack(features)

    def fit(self, X, y):
        X_poly = self._generate_polynomial_features(X)
        self.linear_model.fit(X_poly, y)
        self.cost_history = self.linear_model.cost_history

    def predict(self, X):
        X_poly = self._generate_polynomial_features(X)
        return self.linear_model.predict(X_poly)
