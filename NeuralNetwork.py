import numpy as np

class NeuralNetwork:
    def __init__(self, hidden_size = 4, output_size = 2, learning_rate = 0.01, number_of_epochs = 6000):
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.number_of_epochs = number_of_epochs
        self.input_size = None
        self.W1 = None
        self.W2 = None
        self.b1 = None
        self.b2 = None


    def fit(self, X, y):
        self.input_size = X.shape[1]
        self.W1 = np.random.rand(self.input_size, self.hidden_size)- 0.5
        self.b1 = np.random.rand(1, self.hidden_size) - 0.5
        self.W2 = np.random.rand(self.hidden_size, self.output_size) - 0.5
        self.b2 = np.random.rand(1, self.output_size) - 0.5

        for epoch in range(self.number_of_epochs):
            # Forward pass
            Z1, A1, Z2, A2 = self._forward_propagation(X)

            # Backward pass
            dW1, db1, dW2, db2 = self._backward_propagation(X, y, Z1, A1, Z2, A2)

            # Update parameters
            self._update_parameters(dW1, db1, dW2, db2)

    def predict(self, X):
        _, _, _, A2 = self._forward_propagation(X)
        return np.argmax(A2, axis=1)  # Return class with highest probability

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        return (Z > 0).astype(float)

    # def _sigmoid(self, Z):
    #     return 1/(1+np.exp(-Z))

    def _softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Stability adjustment
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def _forward_propagation(self, X):
        # Input to hidden layer
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._relu(Z1)

        # Hidden to output layer
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._softmax(Z2)
        return Z1, A1, Z2, A2

    def _one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y

    def _backward_propagation(self, X, y, Z1, A1, Z2, A2):
        m = X.shape[0]  # Number of samples

        y_one_hot = self._one_hot(y)
        dZ2 = A2 - y_one_hot
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Gradients for hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._relu_derivative(Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        return dW1, db1, dW2, db2

    def _update_parameters(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2











