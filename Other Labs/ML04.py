#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:15:16 2024

@author: may
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:51:59 2024

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
    for fold in range(0, f):
        print(f"Fold {fold + 1}")
        X_fold = np.row_stack(X_groups[0:fold] + X_groups[fold+1:])
        X_test = X_groups[fold]
        Y_fold = np.concatenate(Y_groups[0:fold] + Y_groups[fold+1:])
        Y_test = Y_groups[fold]
        
        res = np.matmul( np.linalg.inv(np.matmul(X_fold.T, X_fold)), np.matmul(X_fold.T, Y_fold))
        print(f"Coefficients are {res}")
       
        eqs = X_fold * res
        values = np.array(eqs @ np.ones((X_fold.shape[1], 1)))
        values = values.reshape(-1)
        err = (Y_fold - values)
        MSE = np.sum(err * err) / (X_fold.shape[0])
        print(f"Training MSE = {MSE}")
        
        eqs = X_test * res
        values = np.array(eqs @ np.ones((X_test.shape[1], 1)))
        values = values.reshape(-1)
        err = (Y_test - values)
        MSE = np.sum(err * err) / (X_test.shape[0])
        print(f"Testing MSE = {MSE}")
        print("")
        testing_MSEs.append(MSE)
    return testing_MSEs
        
file = "Data/base_kfold_lab_data.csv"
data = pd.read_csv(file)
x = np.array(data[data.columns[0]])
Y = np.array(data[data.columns[-1]])
MSEs = []
folds = 5
for power in range(1, 5):
    print(f"Greatest Power: {power}")
    MSEs.append(MSE(power, x, Y, folds))
    
for fold in range(folds):
    print(f"Fold = {fold}", end=", ")
    for MSEset in MSEs:
        print(f"{MSEset[fold]}", end=", ")
    print()
    
for MSEset in MSEs:
    mse_avg = 0
    for fold in range(folds):
        mse_avg += MSEset[fold]
    mse_avg /= len(MSEs)
    print(mse_avg)
    
#x_curve = np.linspace(0, 100)
#y_curve = res[0] + res[1] * x_curve + res[2] * x_curve * x_curve
#plt.plot(x_curve, y_curve, color="red")
#plt.scatter(x, Y, color="blue")
#plt.xlabel("Speed")
#plt.ylabel("Braking Distance")
#plt.title("Braking Distance vs. Speed")
#plt.show()