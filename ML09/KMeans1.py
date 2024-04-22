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
    num_groups = [np.count_nonzero(groups == idx) for idx in range(int(max(groups)) + 1)]
    new_C = new_C / np.c_[num_groups, num_groups]
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

#print(groups)

colors = ["red", "green", "blue"]

for idx in range(len(X)):
    plt.scatter(X[idx], Y[idx], label=f"Cluster {int(groups[idx] + 1)}", color=colors[int(groups[idx])])

plt.scatter(C[:, 0], C[:, 1], label="Centroids", color="yellow", marker="X", s=100)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()