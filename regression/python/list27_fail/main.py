def f() -> float:
    xs = [3.0, 5.1]
    b = xs.pop()
    a = xs.pop()
    return a + b


assert f() == 8.0
