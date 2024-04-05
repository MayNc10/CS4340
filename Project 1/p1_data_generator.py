#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 5 03:09:00 2024

@author: may
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def graph(x, Y, power_sets, plt, names):
    plt.scatter(x, Y, label="Base Data")
    for (idx, powers) in enumerate(power_sets):
        X = np.column_stack([x**p for p in powers])
        res = np.matmul( np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
        values = X @ res  
        plt.scatter(x, values, label=names[idx])
    plt.legend()
    plt.show()

def MSE(powers, x, y, f):
    MSEs = []
    
    X = np.column_stack([x**p for p in powers])
    k = len(X) // f
    X_groups = [X[range(i, min(k+i, len(X)) ), :] for i in range(0,len(X), k)]
    Y_groups = [y[range(i, min(k+i, len(y)) )] for i in range(0,len(y), k)]
    
    res = np.matmul( np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
    print(f"Coefficients = {res}")

    for fold in range(0, f):
        print(f"Fold {fold + 1}")
        X_fold = np.row_stack(X_groups[0:fold] + X_groups[fold+1:])
        X_test = X_groups[fold]
        Y_fold = np.concatenate(Y_groups[0:fold] + Y_groups[fold+1:])
        Y_test = Y_groups[fold]
        
        res = np.matmul( np.linalg.inv(np.matmul(X_fold.T, X_fold)), np.matmul(X_fold.T, Y_fold))
        print(f"Coefficients are {res}")
       
        values = X_fold @ res        
        err = (Y_fold - values)
        MSE = np.sum(err * err) / (X_fold.shape[0])
        print(f"Training MSE = {MSE}")
        MSEs.append(MSE)
        
        values = X_test @ res
        err = (Y_test - values)
        MSE = np.sum(err * err) / (X_test.shape[0])
        print(f"Testing MSE = {MSE}")
        print("")
        MSEs.append(MSE)
    res = np.matmul( np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
    print(f"Full Data Coefficients are {res}")
    return MSEs
        
file = "Data/Women100M.csv"
data = pd.read_csv(file)
x = np.array(data[data.columns[0]])
Y = np.array(data[data.columns[-1]])

MSEs = [[] for _ in range(12)]
folds = 5
power_sets = [[0, 1], [0, 1, 2], [0, 1, 0.5]]
names = ["Linear", "Quadratic", "Linear + SqRt"]
for power_set in power_sets:
    print(f"Function Powers: {power_set}")
    these_MSEs = MSE(power_set, x, Y, folds)
    for idx in range(len(these_MSEs)):
        MSEs[idx].append(these_MSEs[idx])

graph(x, Y, power_sets, plt, names)

MSEs[-2] = [sum(MSEs[0 : 10 : 2][i] ) for i in range(len(power_sets))]
MSEs[-1] = [sum(MSEs[1 : 11 : 2][i] ) for i in range(len(power_sets))]
df = pd.DataFrame(MSEs, columns=names)
df.to_csv("Project 1/P1_Data.csv", index=False)
    
    
    