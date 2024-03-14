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
    print(f"Accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")

def classify(x, y, Classes, f, KNN):
    
    k = len(x) // f
    X_groups = [x[range(i, k+i)] for i in range(0,len(x), k)]
    Y_groups = [y[range(i, k+i)] for i in range(0,len(y), k)]
    Class_groups = [Classes[range(i, k+i)] for i in range(0,len(Classes), k)]
    
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
            
        print_cm(confusion_matrix)
        print("")

file = "Data/Coyote_ID_data.csv"
data = pd.read_csv(file)
x = np.array(data[data.columns[0]])
y = np.array(data[data.columns[1]])
class_data = np.array(data[data.columns[2]])

F= 5

for KNN in [1, 3, 5, 7, 9]:
    print(f"Results for k={KNN}:")
    classify(x, y, class_data, F, KNN)
    print("")
    
