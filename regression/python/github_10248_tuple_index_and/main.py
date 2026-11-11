# GitHub #10248: a short-circuited out-of-range tuple subscript raises nothing.
def f(i: int) -> int:
    t = (1, 2, 3)
    try:
        b = i < 3 and t[i] == 1
    except IndexError:
        return 7
    assert not b
    return 0


assert f(3) == 0
