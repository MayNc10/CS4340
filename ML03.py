#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:51:59 2024

@author: may
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def MSE(power, x, y, k):
    X = np.column_stack([x**i for  i in range(power + 1)])
    groups = [X[range(i, k+i), :] for i in range(0,           )]

file = "Data/Study_Sleep_v_Exam.csv"
data = pd.read_csv(file)
x = np.array(data[data.columns[0]])
power = 1
X = np.column_stack([x**i for  i in range(power + 1)])
Y = np.array(data[data.columns[-1]])

res = np.matmul( np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
print(f"Coefficients are {res}")

eqs = X * res
values = np.array(eqs @ np.ones((X.shape[1], 1)))
values = values.reshape(-1)
err = (Y - values)
MSE = np.sum(err * err) / (X.shape[0])
print(f"MSE = {MSE}")

#x_curve = np.linspace(0, 100)
#y_curve = res[0] + res[1] * x_curve + res[2] * x_curve * x_curve
#plt.plot(x_curve, y_curve, color="red")
#plt.scatter(x, Y, color="blue")
#plt.xlabel("Speed")
#plt.ylabel("Braking Distance")
#plt.title("Braking Distance vs. Speed")
#plt.show()