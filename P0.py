"""
Reading in and manipulating the Iris Data Set using Pandas Library functions and 
matplotlib.pyplot to plot Sepal Length vs Petal Length
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#What is in this file?
f = input("What is the name of the file? ")
data = pd.read_csv(f)

#Create a table of each type of Iris flowr
Setosa = data[data.species == "setosa"]
Versicolor = data[data.species == "versicolor"]
Virginica = data[data.species == "virginica"]

# sepal_length
# petal_length

while True:
    feature1 = input("What is the first feature you want to plot? ")
    feature2 = input("What is the second feature you want to plot? ")

    #Create scatter plots of sepal length vs petal length
    plt.scatter(Setosa[feature1], Setosa[feature2], 
                marker = "v", c = "red", label = "Setosa")
    plt.scatter(Versicolor[feature1], Versicolor[feature2], 
                marker = "x", c = "green", label = "Versicolor")
    plt.scatter(Virginica[feature1], Virginica[feature2], 
                c = "blue", label = "Virginica")

    #Add plot lables
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(feature1 + " vs " + feature2)
    plt.legend(loc = "upper left")
    plt.plot()
    plt.show()
    
    if input("Do you want to continue? ") not in ["yes", "y", "YES"]:
        break



 