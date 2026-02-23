def swap(a: int, b: int):
    return b, a

a = 2
b = 4
a, b = swap(a, b)

assert a == 4
assert b == 2
