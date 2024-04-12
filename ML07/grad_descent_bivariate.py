import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def standardize_column(x, idx):
    X = np.copy(x)
    m = np.mean(X[:, idx])
    std = np.std(X[:, idx])
    X[:, idx] = (X[:, idx] - m) / std
    return X

def compute_loss(W, X, y):
    return sum([((X[idx] @ W) - y[idx]) ** 2 for idx in range(len(y))]) / (2 * len(y))

def delta_weights(X, y, W):
    Dw = X.T.dot(X.dot(W) - y) / len(y) 
    return Dw

def gradient_descent(X, y, W, learning_rate, iterations):
    for iter in range(iterations):
        deltas = delta_weights(X, y, W)
        W -= learning_rate * deltas
    return W


file = "Data/generated_house_prices.csv"
data = pd.read_csv(file)
x = np.array(data.iloc[:, 0:(len(data.columns) - 1)])
x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
X = standardize_column(standardize_column(x, 1), 2)
y = np.array(data[data.columns[-1]])
biases = np.array([0.0, 0.0, 0.0])

learning_rate = 0.01
iterations = 5000

biases = gradient_descent(X, y, biases, learning_rate, iterations)
print(f"Biases = {biases}")
plt.scatter(x[:, 1], y, label="Data Points", color="blue")
plt.scatter(x[:, 1], X.dot(biases), label="Gradient Descent Fit", color="red", alpha=0.5)
plt.legend()
plt.show()