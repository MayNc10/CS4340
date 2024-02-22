"""
Reading in and manipulating the Iris Data Set using Pandas Library functions and 
matplotlib.pyplot to plot Sepal Length vs Petal Length
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#What is in this file?
f = input("What is the name of the file? ")
data = pd.read_csv(f.strip())

#Create a table of each type of Iris flowr
Setosa = data[data.species == "setosa"]
Versicolor = data[data.species == "versicolor"]
Virginica = data[data.species == "virginica"]

# sepal_length
# petal_length

while True:
    print(data)
    print(data.set_index('species'))
    print("You can do a plot of any two features of the Iris Data set")

    print("The feature codes are:")
    print("   0 = sepal length")
    print("   1 = sepal width")
    print("   2 = petal length")
    print("   3 = petal width")

    feature1 = int(input("Enter feature code for the horizontal axis: "))
    feature2 = int(input("Enter feature code for the vertical axis: "))

    name1 = data.columns[feature1]
    name2 = data.columns[feature2]

    #Create scatter plots of sepal length vs petal length
    plt.scatter(Setosa[name1], Setosa[name2], 
                marker = "v", c = "red", label = "Setosa")
    plt.scatter(Versicolor[name1], Versicolor[name2], 
                marker = "x", c = "green", label = "Versicolor")
    plt.scatter(Virginica[name1], Virginica[name2], 
                c = "blue", label = "Virginica")

    #Add plot lables
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.title(name1 + " vs " + name2)
    plt.legend(loc = "upper right")
    plt.plot()
    plt.show()
    
    if input("Would you like to do another plot? (y/n) ") not in ["yes", "y", "YES"]:
        break



 