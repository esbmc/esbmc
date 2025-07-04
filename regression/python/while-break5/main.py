a: int = 0
b: int = 3

while a < b and b < 10:
    a = a + 1
    if a == 3:
        b = 10  # break the condition

assert a == 3
assert b == 10
