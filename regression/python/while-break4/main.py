x: int = 0
y: int = 0

while x < 4:
    x = x + 1
    if x % 2 == 0:
        continue
    y = y + 1

assert y == 2  # x=1 and x=3

