def f(s: str, k: int):
    if k > 0:
        return s
    elif k < 0:
        return 0
    return 1


a = f("hi", 2)
assert a == "hi"
assert isinstance(a, str)
b = f("hi", -1)
assert b == 0
assert isinstance(b, int)
assert f("hi", 0) == 1
