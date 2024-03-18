#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:38:17 2024

@author: may
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

K = 3

training_file = "Data/data_cm.csv"
training_data = pd.read_csv(training_file)
training_x = np.array(training_data[training_data.columns[0]])
training_y = np.array(training_data[training_data.columns[1]])
training_class = np.array(training_data[training_data.columns[2]])
    
confusion_matrix = np.array([[0, 0], [0, 0]])

for idx in range(len(training_class)):
    sum_squ = (training_x - training_x[idx]) ** 2 + (training_y - training_y[idx]) ** 2
    distances = np.sqrt(sum_squ)
    indices = np.argsort(distances)
    classes = training_class[indices].astype(int)
    votes = np.bincount(classes[:K])
    best_class = np.argmax(votes)
    confusion_matrix[int(training_class[idx])][best_class] += 1
    
print(f"{confusion_matrix}")
accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / len(training_class)
precision = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
f1 = 2 * (precision * recall) / (precision + recall)
print(f"Accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")


