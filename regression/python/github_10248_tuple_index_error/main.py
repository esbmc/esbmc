# GitHub #10248: an out-of-range tuple index raises a catchable IndexError.
def f(i: int) -> int:
    t = (1, 2, 3)
    try:
        return t[i]
    except IndexError:
        return 0


assert f(3) == 0
assert f(-4) == 0
assert f(2) == 3
assert f(-3) == 1
