def total(step: int) -> int:
    t = 0
    for k in range(0, 3, step):
        t = t + k
    return t


try:
    total(0)
    assert False
except ValueError:
    pass
assert total(1) == 3
