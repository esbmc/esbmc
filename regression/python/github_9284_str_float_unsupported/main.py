def g(x):
    if x == "d":
        return "e"
    return x


assert g(1.5) == 1.5
assert g("d") == "e"
