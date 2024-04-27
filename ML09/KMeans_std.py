import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def standardize_column(x, idx):
    X = np.copy(x)
    m = np.mean(X[:, idx])
    std = np.std(X[:, idx])
    X[:, idx] = (X[:, idx] - m) / std
    return X

def kpp(X, C):
    # Add one more center
    D = np.zeros(len(X))
    for idx in range(len(X)):
        coord = X[idx]
        D[idx] = np.min(np.linalg.norm(C - coord, axis=1))
    D /= np.sum(D)
    p = np.random.random()
    idx = 0
    while p >= 0:
        p -= D[idx]
        idx += 1
    idx -= 1
    return np.vstack((C, X[idx]))

def step_centroids(X, C):
    new_C = np.zeros(C.shape)
    groups = np.zeros(X[:, 0].shape)
    for idx in range(len(X)):
        coord  = X[idx]
        dist_C = np.array([np.linalg.norm(ci - coord) for ci in C])
        minpos = np.argmin(dist_C)
        new_C[minpos] += coord
        groups[idx] = minpos
    num_groups = [np.count_nonzero(groups == idx) for idx in range(int(max(groups)) + 1)]
    new_C = new_C / np.c_[num_groups, num_groups]
    return (new_C, groups)

def k_means(X):
    k = 3
    C = np.array([X[np.random.randint(0, len(X))]])
    # Use K++
    while len(C) < k:
        C = kpp(X, C)

    groups = np.zeros(X.shape)

    while True:
        (C, new_groups) = step_centroids(X, C)
        if np.array_equal(groups, new_groups):
            break
        groups = new_groups

    colors = ["red", "green", "blue"]

    for idx in range(len(X)):
        plt.scatter(X[idx, 0], X[idx, 1], label=f"Cluster {int(groups[idx] + 1)}", color=colors[int(groups[idx])])

    plt.scatter(C[:, 0], C[:, 1], label="Centroids", color="yellow", marker="X", s=100)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

    

    

file = "Data/iris_data_two_features.csv"
data = pd.read_csv(file)
X = np.array(data[data.columns[:2]])
k_means(X)
X_std = standardize_column(standardize_column(X, 0), 1)
k_means(X_std)

