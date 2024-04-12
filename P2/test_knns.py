#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:38:17 2024

@author: may
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def print_cm(con_mat):
    #print(f"{con_mat}")
    accuracy = (con_mat[0][0] + con_mat[1][1]) / (con_mat[0][0] + con_mat[1][1] + con_mat[1][0] + con_mat[0][1]) 
    precision = con_mat[1][1] / (con_mat[1][1] + con_mat[0][1])
    recall = con_mat[1][1] / (con_mat[1][1] + con_mat[1][0])
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

def classify(x, y, Classes, f, KNN):
    
    k = len(x) // f
    X_groups = [x[range(i, k+i)] for i in range(0,len(x), k)]
    Y_groups = [y[range(i, k+i)] for i in range(0,len(y), k)]
    Class_groups = [Classes[range(i, k+i)] for i in range(0,len(Classes), k)]
    
    mis_ids = []

    for fold in range(0, f):
        confusion_matrix = np.array([[0, 0], [0, 0]])
        
        print(f"Fold {fold + 1}")
        X_fold = np.concatenate(X_groups[0:fold] + X_groups[fold+1:])
        X_test = X_groups[fold]
        Y_fold = np.concatenate(Y_groups[0:fold] + Y_groups[fold+1:])
        Y_test = Y_groups[fold]
        Class_fold = np.concatenate(Class_groups[0:fold] + Class_groups[fold+1:])
        Class_test = Class_groups[fold]
        
        for idx in range(len(Class_test)):
            sum_squ = (X_fold - X_test[idx]) ** 2 + (Y_fold - Y_test[idx]) ** 2
            distances = np.sqrt(sum_squ)
            indices = np.argsort(distances)
            classes = Class_fold[indices].astype(int)
            votes = np.bincount(classes[:KNN])
            best_class = np.argmax(votes)
            confusion_matrix[int(Class_test[idx])][best_class] += 1
            
        cm = confusion_matrix
        mis_ids.append(confusion_matrix[1][0] + confusion_matrix[0][1])
        print_cm(cm)
        print("")

    return mis_ids

def classify_full_set(x, y, Classes, KNN):
    confusion_matrix = np.array([[0, 0], [0, 0]])

    for idx in range(len(Classes)):
        sum_squ = (x - x[idx]) ** 2 + (y - y[idx]) ** 2
        distances = np.sqrt(sum_squ)
        indices = np.argsort(distances)
        classes = Classes[indices].astype(int)
        votes = np.bincount(classes[:KNN])
        best_class = np.argmax(votes)
        confusion_matrix[int(Classes[idx])][best_class] += 1
        
    return confusion_matrix

def classify_test(x, y, classes, X, Y, Classes, KNN):
    confusion_matrix = np.array([[0, 0], [0, 0]])

    for idx in range(len(Classes)):
        sum_squ = (x - X[idx]) ** 2 + (y - Y[idx]) ** 2
        distances = np.sqrt(sum_squ)
        indices = np.argsort(distances)
        classes_chosen = classes[indices].astype(int)
        votes = np.bincount(classes_chosen[:KNN])
        best_class = np.argmax(votes)
        confusion_matrix[int(Classes[idx])][best_class] += 1
        
    return confusion_matrix

file = "P2/P2train.csv"
data = pd.read_csv(file)
x = np.array(data[data.columns[0]])
y = np.array(data[data.columns[1]])
class_data = np.array(data[data.columns[2]])

testing_file = "P2/P2test.csv"
testing_data = pd.read_csv(testing_file)
x_test = np.array(testing_data[testing_data.columns[0]])
y_test = np.array(testing_data[testing_data.columns[1]])
class_data_test = np.array(testing_data[testing_data.columns[2]])

F = 5

cms = []
colors = ["red", "blue", "green", "orange", "purple"]
KNNs = [3, 5, 7, 9, 11, 13, 15]

for KNN in KNNs:
    print(f"Results for k={KNN}:")
    mis_ids = classify(x, y, class_data, F, KNN)
    for (idx, id) in enumerate(mis_ids):
        plt.scatter(KNN, id, label=f"Fold {idx + 1}" if KNN == KNNs[0] else "", color=colors[idx], alpha=0.5)
    print("Results for all training data:")
    cm = classify_full_set(x, y, class_data, KNN)
    print_cm(cm)
    cms.append(cm)
    print("Results for all testing data:")
    cm = classify_test(x, y, class_data, x_test, y_test, class_data_test, KNN)
    print_cm(cm)
    
plt.title("Misidentifications per KNN")
plt.xlabel("KNN")
plt.xlabel("Number of Misidentifications")
plt.legend()
plt.show()

for (idx, cm) in enumerate(cms):
    KNN = KNNs[0] + 2 * idx
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]) 
    plt.scatter(KNN, accuracy)

plt.title("Accuracy per KNN")
plt.xlabel("KNN")
plt.xlabel("Average Accuracy")
plt.show()

