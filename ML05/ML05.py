#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:38:17 2024

@author: may
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

K = 5

training_file = "Data/apples_oranges.csv"
training_data = pd.read_csv(training_file)
training_x = np.array(training_data[training_data.columns[0]])
training_y = np.array(training_data[training_data.columns[1]])
training_class = np.array(training_data[training_data.columns[2]])

testing_file = "Data/test_apples_oranges2.csv"
testing_data = pd.read_csv(testing_file)
testing_x = np.array(testing_data[testing_data.columns[-2]])
testing_y = np.array(testing_data[testing_data.columns[-1]])

for idx in range(len(training_class)):
    color = "red" if training_class[idx] == 0 else "orange"
    label = "Apples" if training_class[idx] == 0 else "Oranges"
    plt.scatter(training_x[idx], training_y[idx], color=color, label=label)
    
for idx in range(len(testing_x)):
    sum_squ = (training_x - testing_x[idx]) ** 2 + (training_y - testing_y[idx]) ** 2
    distances = np.sqrt(sum_squ)
    indices = np.argsort(distances)
    classes = training_class[indices].astype(int)
    votes = np.bincount(classes[:K])
    best_class = np.argmax(votes)
    
    color = "green" if best_class == 0 else "blue"
    label = "Apple Test points" if best_class == 0 else "Orange Test Points"
    plt.scatter(testing_x[idx], testing_y[idx], color=color, label=label)
    

plt.xlabel("Sweetness")
plt.ylabel("Crunchiness")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()



