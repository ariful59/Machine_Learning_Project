import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, number__of_iters=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.number_of_iteration = number__of_iters
        self.weights, self.bias = None, None
        self.tolerance = tolerance

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.number_of_iteration):
            linear_model = np.dot(X, self.weights) + self.bias  # y = wx + b
            logistic_regression = 1 / (1 + np.exp(-linear_model))
            dw = (1 / n_samples) * np.dot(X.T, (logistic_regression - y)) #1/N*2x(y_predict - y)
            db = (1 / n_samples) * np.sum(logistic_regression - y)  # #1/N*2(y_predict - y)
            self.weights -= self.learning_rate * dw #Update w
            self.bias -= self.learning_rate * db # update b
            #loop this until we reached local minimum
            if np.linalg.norm(dw) < self.tolerance and abs(db) < self.tolerance:
                print(f"Converged after {i + 1} iterations")
                break

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        logistic_regression = 1 / (1 + np.exp(-linear_model))
        y_pred = [0  if y <= 0.5 else 1 for y in logistic_regression]
        return y_pred
