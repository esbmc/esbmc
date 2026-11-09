calls = [0]


def g() -> int:
    calls[0] += 1
    return -5


assert abs(g()) == 5
assert calls[0] == 1
