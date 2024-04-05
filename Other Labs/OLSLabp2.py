#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:58:47 2024

@author: may
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:31:51 2024

@author: may
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file = input("Enter the name of the file: ")
data = pd.read_csv(file)
x = np.array(data[data.columns[0]])
y = np.array(data[data.columns[1]])

x_avg = np.mean(x)
y_avg = np.mean(y)

x_diff = x - x_avg
y_diff = y - y_avg

m = np.sum(x_diff * y_diff)/np.sum(x_diff * x_diff)
b = y_avg - m * x_avg

err = y - (m * x + b)
MSE = np.sum(err * err) / len(x)
print(f"m: {m}, b: {b}, MSE: {MSE}")

X = np.linspace(50, 85)
y_line = m * X + b

plt.plot(X, y_line, color='red')
plt.scatter(x, y, color='blue')
plt.xlabel(data[data.columns[0]].name )
plt.ylabel(data[data.columns[1]].name )
plt.title(data[data.columns[1]].name + " vs " + data[data.columns[0]].name)

plt.show()
