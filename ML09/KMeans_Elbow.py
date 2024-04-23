import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import pandas as pd

def step_centroids(X, Y, C, k):
    new_C = np.zeros(C.shape)
    groups = np.zeros(X.shape)
    for idx in range(len(X)):
        coord  = np.array([X[idx], Y[idx]])
        dist_C = np.array([np.linalg.norm(ci - coord) for ci in C])
        minpos = np.argmin(dist_C)
        new_C[minpos] += coord
        groups[idx] = minpos
    num_groups = [np.count_nonzero(groups == idx) for idx in range(k)]
    new_C = new_C / np.c_[num_groups, num_groups]
    return (new_C, groups)

def find_centroids(X, Y, k):
    C = np.array([np.array([X[idx], Y[idx]]) for idx in np.random.randint(0, len(X), k)])
    groups = np.zeros(X.shape)
    diff_history = []

    while True:
        (C, new_groups) = step_centroids(X, Y, C, k)
        diff = np.sum(new_groups - groups)
        if diff in diff_history:
            # restart with new seeds, this is giving too many errors
            return find_centroids(X, Y, k)
        diff_history.append(diff)
        if np.array_equal(groups, new_groups) or diff == 0:
            break
        groups = new_groups

    return C

file = "Data/data_points.csv"
data = pd.read_csv(file)
X = np.array(data[data.columns[0]])
Y = np.array(data[data.columns[1]])

KMAX = 10
reps = 10
SSEs = np.array([0 for _ in range(1, KMAX + 1)])
for k in range(1, KMAX + 1):
    this_sses = []
    for rep in range(reps):
        print(f"K: {k}, rep: {rep}")
        C = find_centroids(X, Y, k)
        # calculate SSE
        sse = 0
        for idx in range(len(X)):
            coord  = np.array([X[idx], Y[idx]])
            dist_C = np.array([np.linalg.norm(ci - coord) for ci in C])
            min_dist = np.min(dist_C)
            sse += min_dist ** 2
        this_sses.append(sse)
    # min or average?
    SSEs[k - 1] = np.min(this_sses)

ks = np.array([idx for idx in range(1, KMAX + 1)])
plt.plot(ks, SSEs, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.xticks(ks)
plt.grid()
plt.ylabel("WCSS")
plt.title("The Elbow Method using Custom K-means Clustering")
plt.show()
    





