def f():
    p = (["a", "b"], 3)
    (x, y), m = p
    return y if m == 3 else x


assert f() == "b"
