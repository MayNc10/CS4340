import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

main_folder = True
path_prepend = "Project 4/" if main_folder else ""

def print_cm(con_mat):
    #print(f"{con_mat}")
    accuracy = (con_mat[0][0] + con_mat[1][1]) / (con_mat[0][0] + con_mat[1][1] + con_mat[1][0] + con_mat[0][1]) 
    precision = con_mat[1][1] / (con_mat[1][1] + con_mat[0][1])
    recall = con_mat[1][1] / (con_mat[1][1] + con_mat[1][0])
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"TP: {con_mat[1][1]}, TN: {con_mat[0][0]}, FP: {con_mat[0][1]}, FN: {con_mat[1][0]} Accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")

def sigmoid(X, W):
    return 1 / (1 + np.exp(-1 * X.dot(W)))

file = input("Please enter a path to a testing file: ")
data = pd.read_csv(file)
x = np.array(data.iloc[:, 0:(len(data.columns) - 1)])
x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
X = x
y = np.array(data[data.columns[-1]])
W = []
fin = open(input("Please enter a path to a weights file: "))
for line in fin.readlines():
    W.append(float(line))

W = np.array(W)

P = np.round(sigmoid(X, W))
confusion_matrix = np.array([[0,0], [0,0]])
for idx in range(len(y)):
    confusion_matrix[int(P[idx])][int(y[idx])] += 1

print_cm(confusion_matrix)