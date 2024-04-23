import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def compute_loss(W, X, y):
    return sum([np.log(X[idx] @ W) * y[idx] + (1 - y[idx]) * (np.log(1 - X[idx] @ W)) for idx in range(len(y))]) / len(y)

def delta_weights(X, y, W):
    p = 1 / (1 + np.exp(-1 * X.dot(W)))
    Dw = (p - y).dot(X)
    return Dw

def gradient_descent(X, y, W, learning_rate, iterations):
    for iter in range(iterations):
        deltas = delta_weights(X, y, W)
        W -= learning_rate * deltas
    return W


file = "Data/Logistic_Regression_Data2.csv"
data = pd.read_csv(file)
x = np.array(data.iloc[:, 0:(len(data.columns) - 1)])
x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
#X = standardize_column(standardize_column(x, 1), 2)
y = np.array(data[data.columns[-1]])
W = np.array([0.0, 0.0])

learning_rate = 0.01
iterations = 5000

W = gradient_descent(x, y, W, learning_rate, iterations)
print(f"Weights = {W}")
P = np.round(1 / (1 + np.exp(-1 * x.dot(W))))
print("Can't plot data points for only one variable!")
confusion_matrix = np.array([[0,0], [0,0]])
for idx in range(len(y)):
    confusion_matrix[int(P[idx])][int(y[idx])] += 1

print_cm(confusion_matrix)