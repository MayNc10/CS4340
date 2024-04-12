import numpy as np
import matplotlib.pyplot as plt

MPL = 100
LR = 0.3
HPT = 20

def learning(t):
    return MPL / (1 + np.exp(-LR * (t - HPT)))

print(f"Learning at 10 sessions = {learning(10)}, 20 sessions = {learning(20)}, 40 sessions = {learning(40)}")

X = np.linspace(0, 40, 100)
Y = learning(X)
plt.scatter(X, Y, label="Learning Curve")
plt.xlabel("Sessions")
plt.ylabel("Proficiency")
plt.legend()
plt.show()