a = 1/3
r = 0.5
max_iter = 30
val = 0

for iter in range(max_iter):
    val += a * (r ** iter)

print(f"Value after {max_iter} iterations: {val}")