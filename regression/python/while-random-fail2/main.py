from random import randint

n: int = randint(0, 5)
x: int = 10

while n > 0:
    x = x - 3
    n = n - 1

# If n == 5, x = 10 - 15 = -5
assert x >= 0  # FAILS for large n
