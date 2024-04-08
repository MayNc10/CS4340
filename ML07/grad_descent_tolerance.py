import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def compute_loss(w0, w1, X, y):
    return sum([((w0 + w1 * X[idx]) - y[idx]) ** 2 for idx in range(len(y))]) / (2 * len(y))

def delta_weights(X, y, w0, w1):
    Dw0 = sum([(w0 + w1 * X[idx]) - y[idx] for idx in range(len(y))]) / len(y)
    Dw1 = sum([((w0 + w1 * X[idx]) - y[idx]) * X[idx] for idx in range(len(y))]) / len(y)
    return np.array([Dw0, Dw1])

def gradient_descent(X, y, w0, w1, learning_rate, tolerance):
    biases = np.array([w0, w1])
    iters = 0
    while True:
        iters += 1
        before_loss = compute_loss(biases[0], biases[1], X, y)
        deltas = delta_weights(X, y, biases[0], biases[1])
        biases -= learning_rate * deltas
        after_loss = compute_loss(biases[0], biases[1], X, y)
        if abs(after_loss - before_loss) < tolerance:
            break

    print(f"Iterations: {iters}")
    return biases


file = "Data/grad_desc_exam_scores.csv"
data = pd.read_csv(file)
x = np.array(data[data.columns[0]])
y = np.array(data[data.columns[1]])
biases = np.array([0.0, 0.0])

learning_rate = 0.01
tolerance = 0.0001
X = np.linspace(min(x), max(x), num=50)

biases = gradient_descent(x, y, biases[0], biases[1], learning_rate, tolerance)
print(f"Biases = {biases}")
plt.scatter(x, y, label="Data Points", color="blue")
plt.plot(X, X * biases[1] + biases[0], label="Gradient Descent Fit", color="red")
plt.legend()
plt.show()