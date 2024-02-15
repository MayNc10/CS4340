#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:31:51 2024

@author: may
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("Data/Height_ShoeSize.csv")
x = np.array(data['Height'])
y = np.array(data['Shoe_Size'])

x_avg = np.mean(x)
y_avg = np.mean(y)

x_diff = x - x_avg
y_diff = y - y_avg

m = np.sum(x_diff * y_diff)/np.sum(x_diff * x_diff)
b = y_avg - m * x_avg

err = y - (m * x + b)
MSE = np.sum(err * err)
print(f"m: {m}, b: {b}, MSE: {MSE}")

X = np.linspace(50, 85)
y_line = m * X + b

plt.plot(X, y_line, color='red')
plt.scatter(x, y, color='blue')
plt.xlabel("Height")
plt.ylabel("Shoe Size")
plt.title("Shoe Size vs. Height")

plt.show()
