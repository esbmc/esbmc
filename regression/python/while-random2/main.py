from random import randint

a: int = 0
b: int = randint(3, 6)

while a < b and b < 10:
    a = a + 1
    if a == 3:
        b = 10

assert a <= 3
assert b == 10 or b < 10
