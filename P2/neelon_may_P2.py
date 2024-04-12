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
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

color_list_training = ["orange", "blue"]
color_list = ["red", "green"]
name_list = ["Failed QC", "Passed QC"]

K = 3

training_file = input("Please enter the name of the training data file: ")
training_data = pd.read_csv(training_file)
training_x = np.array(training_data[training_data.columns[0]])
training_y = np.array(training_data[training_data.columns[1]])
training_class = np.array(training_data[training_data.columns[2]])

testing_file = input("Please enter the name of the testing data file: ")
testing_data = pd.read_csv(testing_file)
testing_x = np.array(testing_data[testing_data.columns[0]])
testing_y = np.array(testing_data[testing_data.columns[1]])
testing_class = np.array(testing_data[testing_data.columns[2]])

for idx in range(len(training_class)):
    color = color_list_training[training_class[idx]]
    label = name_list[training_class[idx]]
    plt.scatter(training_x[idx], training_y[idx], color=color, label=label, marker='^')

confusion_matrix = np.array([[0, 0], [0, 0]])

for idx in range(len(testing_x)):
    sum_squ = (training_x - testing_x[idx]) ** 2 + (training_y - testing_y[idx]) ** 2
    distances = np.sqrt(sum_squ)
    indices = np.argsort(distances)
    classes = training_class[indices].astype(int)
    votes = np.bincount(classes[:K])
    best_class = np.argmax(votes)
    confusion_matrix[int(testing_class[idx])][best_class] += 1
    
    color = color_list[best_class]
    label = name_list[best_class] + " Test Points"
    plt.scatter(testing_x[idx], testing_y[idx], color=color, label=label, marker='x' if best_class == 0 else "o")
    
print("Confusion Matrix:")
print_cm(confusion_matrix)

plt.xlabel(training_data.columns[0])
plt.ylabel(training_data.columns[1])

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()