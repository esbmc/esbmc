def k(x, y):
    if y == "a":
        return 1
    return y


assert k(1, "a") == 1
assert k("b", 2) == 2
