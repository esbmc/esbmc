# GitHub #10248: the IndexError guard of a tuple subscript in a list literal
# must not run ahead of the elements before it.
def f(i: int, z: int) -> int:
    t = (1, 2, 3)
    try:
        xs = [1 // z, t[i]]
    except ZeroDivisionError:
        return 1
    except IndexError:
        return 2
    return xs[1]


assert f(3, 0) == 2
