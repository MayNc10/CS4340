a = 1/3
r = 0.5
val = 0
tolerance = 1E-6
iter = 0
while True:
    next_val = a * (r ** iter)
    val += next_val
    if abs(next_val) < tolerance:
        break
    iter += 1

print(f"Value after {iter} iterations with tolerance {tolerance}: {val}")
