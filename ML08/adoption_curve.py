import numpy as np
import matplotlib.pyplot as plt

P = 10000
A = 0.2
H = 6

def adoption(t):
    return P / (1 + np.exp(-A * (t - H)))

print(f"Adoption at 3 months = {adoption(3)}, 6 months = {adoption(6)}, 12 months = {adoption(12)}")

X = np.linspace(0, 12, 100)
Y = adoption(X)
plt.scatter(X, Y, label="Adoption Curve")
plt.xlabel("Months")
plt.ylabel("Adoption")
plt.legend()
plt.show()