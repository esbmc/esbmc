x: int = 0
n: int = 0
while x != 0 and 12 // x > 2:
    x -= 1
    n += 1
assert n == 0

x = 3
while x != 0 and 12 // x > 2:
    x -= 1
    n += 1
assert n == 3
