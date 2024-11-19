import numpy as np

class SupportVectorClassifier:
    def __init__(self, learning_rate =0.01, lamda_value = 0.01, n_iteration = 1000):
        self.learning_rate = learning_rate
        self.lamda_value = lamda_value
        self.n_iteration = n_iteration
        self.weights, self.bias = None, None

    def fit(self, X, y):
        #converted all the value to either -1 or 1 depending on the condition.
        modified_y = np.where(y<=0, -1, 1)
        #Take the attribute numbers (row) and number of features (col)
        n_samples, n_features = X.shape
        #create a default weights (we can also take the random sample value for default weights also)
        self.weights = np.zeros(n_features)
        self.bias = 0
        #fixed number of time iterate
        for _ in range(self.n_iteration):
            #for every feature in the X
            for index, x in enumerate(X):
                # y_i * w* x_i - b
                svc = modified_y[index] * (np.dot(X[index], self.weights) - self.bias)
                if svc >= 1:
                    # w = w - lr * 2 * (\lamda) * weight
                    self.weights -= self.learning_rate * (2 * self.lamda_value * self.weights)
                else:
                    # w = w - lr * (2 * (\lamda) * weight - x_i * m_y_i)
                    # b = b - lr * m_y_i
                    self.weights -= self.learning_rate * 2 * self.lamda_value * self.weights - np.dot(x, modified_y[index])
                    self.bias -= self.learning_rate * modified_y[index]

    def predict(self,X):
        svc = np.dot(X, self.weights) - self.bias
        # just return the sign where it is positive or negative
        y_hat = np.where(svc <= -1, 0, 1)
        return y_hat
