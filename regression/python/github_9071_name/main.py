def f(c: bool) -> None:
    x = 0
    if c:
        x = "x"
    assert x == "x"
    assert isinstance(x, str)


x = 0
flag = True
if flag:
    x = "x"
else:
    x = "y"
assert x == "x"
f(True)
