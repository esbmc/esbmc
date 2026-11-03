g = 1


def boom() -> int:
    global g
    g = 10
    raise ValueError("stop")


try:
    x = g + boom()
except ValueError:
    x = g
assert x == 10
