def f(x: int):
    if x > 0:
        return x
    return "neg"

y = f(5) + 1
assert y == 6
