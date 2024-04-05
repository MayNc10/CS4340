val = 0
tolerance = 1E-6
iter = 0
while True:
    next_val = (1 if iter % 2 == 0 else -1) / (1 + 2 * iter)
    val += next_val
    if abs(next_val) < tolerance:
        break
    iter += 1

val *= 4

print(f"Value after {iter} iterations with tolerance {tolerance}: {val}")
