def func(x="d"):
    if x == "d":
        return "a"
    return x


assert func(5) == 6
assert func() == "a"
