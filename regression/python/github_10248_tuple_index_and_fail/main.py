# GitHub #10248: the IndexError guard of a short-circuited tuple subscript
# must not run ahead of the `and`, where the handler would swallow it.
def f(i: int) -> int:
    t = (1, 2, 3)
    try:
        b = i < 3 and t[i] == 1
    except IndexError:
        return 7
    assert b
    return 0


f(3)
