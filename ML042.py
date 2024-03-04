#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 1 13:15:16 2024

@author: may
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def MSE(power, x, y, f):
    testing_MSEs = []
    
    X = np.column_stack([x**i for  i in range(power + 1)])
    k = len(X) // f
    X_groups = [X[range(i, k+i), :] for i in range(0,len(X), k)]
    Y_groups = [y[range(i, k+i)] for i in range(0,len(y), k)]
    
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
        
        values = X_test @ res
        err = (Y_test - values)
        MSE = np.sum(err * err) / (X_test.shape[0])
        print(f"Testing MSE = {MSE}")
        print("")
        testing_MSEs.append(MSE)
    return testing_MSEs
        
file = "Data/building_energy_consumption.csv"
data = pd.read_csv(file)
x = np.array(data[data.columns[0]])
Y = np.array(data[data.columns[-1]])

MSEs = []
folds = 5
for power in range(1, 4):
    print(f"Degree: {power}")
    MSEs.append(MSE(power, x, Y, folds))
    print(f"Average MSE for {power} = {sum(MSEs[-1]) / len( MSEs[-1]) }")
    
    
    