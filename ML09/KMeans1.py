import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def step_centroids(X, Y, C):
    new_C = np.zeros(C.shape)
    groups = np.zeros(X.shape)
    for idx in range(len(X)):
        coord  = np.array([X[idx], Y[idx]])
        dist_C = np.array([np.linalg.norm(ci - coord) for ci in C])
        minpos = np.argmin(dist_C)
        new_C[minpos] += coord
        groups[idx] = minpos
    new_C /= len(X)
    return (new_C, groups)

file = "Data/clustered_data.csv"
data = pd.read_csv(file)
X = np.array(data[data.columns[0]])
Y = np.array(data[data.columns[1]])
C = np.array([[2.7, 3.36], [2.42, 2.53], [2.64, 10.15]])
groups = np.zeros(X.shape)

while True:
    (C, new_groups) = step_centroids(X, Y, C)
    if np.array_equal(groups, new_groups):
        break
    groups = new_groups

for idx in range(len(X)):
    plt.scatter(X[idx], Y[idx], label=f"Cluster {int(groups[idx] + 1)}")

plt.scatter(C[:, 0], C[:, 1], label="Centroids")
plt.legend()
plt.show()