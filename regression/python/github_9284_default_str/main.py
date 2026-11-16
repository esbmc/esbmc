def func(x="d"):
    if x == "d":
        return "a"
    return x


assert func(5) == 5
assert func() == "a"
