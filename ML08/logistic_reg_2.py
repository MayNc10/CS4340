import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def sigmoid(X, W):
    return 1 / (1 + np.exp(-1 * X.dot(W)))

def print_cm(con_mat):
    #print(f"{con_mat}")
    accuracy = (con_mat[0][0] + con_mat[1][1]) / (con_mat[0][0] + con_mat[1][1] + con_mat[1][0] + con_mat[0][1]) 
    precision = con_mat[1][1] / (con_mat[1][1] + con_mat[0][1])
    recall = con_mat[1][1] / (con_mat[1][1] + con_mat[1][0])
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"TP: {con_mat[1][1]}, TN: {con_mat[0][0]}, FP: {con_mat[0][1]}, FN: {con_mat[1][0]} Accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")

def standardize_column(x, idx):
    X = np.copy(x)
    m = np.mean(X[:, idx])
    std = np.std(X[:, idx])
    X[:, idx] = (X[:, idx] - m) / std
    return X

def compute_loss(X, y, W):
    y_hat = sigmoid(X, W)
    return -1 * sum([y[idx] * np.log(y_hat[idx]) + (1 - y[idx]) * (np.log(1 - y_hat[idx])) for idx in range(len(y))]) / len(y)

def delta_weights(X, y, W):
    p = sigmoid(X, W)
    Dw = X.T.dot(p - y) / len(y) 
    return Dw

def gradient_descent(X, y, W, learning_rate, iterations):
    loss_per = []
    for iter in range(iterations):
        deltas = delta_weights(X, y, W)
        W -= learning_rate * deltas
        loss = compute_loss(X, y, W)
        loss_per.append([iter, loss])
    return (W, np.array(loss_per))


file = "Data/logistic_regression_data_standard.csv"
data = pd.read_csv(file)
x = np.array(data.iloc[:, 0:(len(data.columns) - 1)])
x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
#X = standardize_column(standardize_column(x, 1), 2)
y = np.array(data[data.columns[-1]])
W = np.array([0.0 for _ in range(len(x[0, :]))])

learning_rate = 0.01
iterations = 1000

(W, loss_per) = gradient_descent(x, y, W, learning_rate, iterations)
print(f"Weights = {W}")
P = np.round(sigmoid(x, W))
confusion_matrix = np.array([[0,0], [0,0]])
for idx in range(len(y)):
    confusion_matrix[int(P[idx])][int(y[idx])] += 1

print_cm(confusion_matrix)

plt.plot(loss_per[:, 0], loss_per[:, 1], label="Log Loss", color="blue")
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("Logistic Regression Training")
plt.legend()
plt.show()

colormap = np.array(["blue", "orange"])
plt.scatter(x[:, 1], x[:, 2], label="Data", c=colormap[y.astype(int)])
X = np.array([min(x[:, 1]), max(x[:, 1])])
c = -W[0] / W[2]
m = -W[1] / W[2]
X1 = m * X + c
plt.plot(X, X1, label="Decision boundary", lw=1, ls='--')
plt.fill_between(X, X1, min(x[:, 2]), color='tab:blue', alpha=0.2)
plt.fill_between(X, X1, max(x[:, 2]), color='tab:orange', alpha=0.2)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.show()