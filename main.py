import numpy as np
import pandas as pd

# Hàm sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Đạo hàm của hàm sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Lớp neural network
class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # Mô hình layer với độ dài [2,2,1]
        self.layers = layers
        # Hệ số learning rate
        self.alpha = alpha
        # Tham số W, b
        self.W = []
        self.b = []

        # Khởi tạo các tham số cho mỗi layer
        for i in range(0, len(layers) - 1):
            w_ = np.random.randn(layers[i], layers[i + 1])
            b_ = np.zeros((layers[i + 1], 1))
            self.W.append(w_ / layers[i])
            self.b.append(b_)

    # Tạm từ mô hình neural network
    def __repr__(self):
        return "Neural network [{}]".format("-".join(str(l) for l in self.layers))

    # Train mô hình với dữ liệu
    def fit_partial(self, x, y):
        A = [x]
        # Quá trình feedforward
        out = A[-1]
        for i in range(0, len(self.layers) - 1):
            out = sigmoid(np.dot(out, self.W[i]) + self.b[i].T)
            A.append(out)
            print(f"Layer {i+1} output: {A[-1]}")  # Debugging statement

        # Quá trình backpropagation
        y = y.reshape(-1, 1)
        dA = [-(y / A[-1] - (1 - y) / (1 - A[-1]))]
        dW = []
        db = []
        for i in reversed(range(0, len(self.layers) - 1)):
            dw_ = np.dot(A[i].T, dA[-1] * sigmoid_derivative(A[i + 1]))
            db_ = (np.sum(dA[-1] * sigmoid_derivative(A[i + 1]), axis=0)).reshape(-1, 1)
            dA_ = np.dot(dA[-1] * sigmoid_derivative(A[i + 1]), self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)
            print(f"Layer {i+1} dW: {dW[-1]}, db: {db[-1]}")  # Debugging statement

        # Đảo ngược dW, db
        dW = dW[::-1]
        db = db[::-1]

        # Gradient descent
        for i in range(0, len(self.layers) - 1):
            self.W[i] = self.W[i] - self.alpha * dW[i]
            self.b[i] = self.b[i] - self.alpha * db[i]

    def fit(self, X, y, epochs=20, verbose=10):
        for epoch in range(0, epochs):
            self.fit_partial(X, y)
            if epoch % verbose == 0:
                loss = self.calculate_loss(X, y)
                print("Epoch {}, loss {}".format(epoch, loss))

    # Dự đoán
    def predict(self, X):
        for i in range(0, len(self.layers) - 1):
            X = sigmoid(np.dot(X, self.W[i]) + self.b[i].T)
        return X

    # Tính loss function
    def calculate_loss(self, X, y):
        y_predict = self.predict(X)
        return -(np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict)))

# Example data
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
y_train = np.array([0, 1, 1, 0])  # Target labels (for XOR problem)

# Initialize and train the neural network
nn = NeuralNetwork([2, 2, 1], alpha=0.1)
nn.fit(X_train, y_train, epochs=10000, verbose=1000)

# Make predictions
predictions = nn.predict(X_train)
print("Predictions:", predictions)
