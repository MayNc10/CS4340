#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:51:59 2024

@author: may
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file = "Data/advertising_sales_data.csv"
data = pd.read_csv(file)
x = np.array(data.iloc[:, 0:3])
print(x)
#x = x.reshape(x.shape[0], 1)
X = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
print(X)
Y = np.array(data[data.columns[3]])

res = np.matmul( np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
print(res)
