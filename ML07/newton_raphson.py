f = lambda x : x ** 2 - 2
f_prime = lambda x : 2 * x

val = 1.5
tolerance = 1E-6
iter = 0
while True:
    change = f(val) / f_prime(val)
    val -= change
    if abs(change) < tolerance:
        break
    iter += 1


print(f"Value after {iter} iterations with tolerance {tolerance}: {val}")