from random import randint

x: int = 0
flag: bool = True
limit: int = randint(2, 5)

while x < limit and flag:
    x = x + 1
    if x == 2:
        flag = False

assert x <= limit
assert x >= 1
