import numpy as np
from src.Linear_Regression import LinearRegression

class PolynomialRegressionScratch:
    def __init__(self, degree=2, learning_rate=0.05, iterations=2000):
        self.degree = degree
        self.linear_model = LinearRegression(learning_rate=learning_rate, iterations=iterations)
        
    def _generate_polynomial_features(self, X):
        """
        Generates polynomial and interaction features up to the specified degree (e.g., degree=2).
        For X = [x1, x2], degree=2 generates: [x1, x2, x1^2, x2^2, x1*x2]
        """
        m, n = X.shape
        features = [X]
        
        # Add powers and interactions
        if self.degree >= 2:
            # Squared terms: x_i^2
            X_squared = X ** 2
            features.append(X_squared)
            
            # Interaction terms: x_i * x_j (for i < j)
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
        # Note: Standardization should ideally happen AFTER generating poly features, 
        # or we can train the underlying linear model on poly features.
        self.linear_model.fit(X_poly, y)
        self.cost_history = self.linear_model.cost_history

    def predict(self, X):
        X_poly = self._generate_polynomial_features(X)
        return self.linear_model.predict(X_poly)
