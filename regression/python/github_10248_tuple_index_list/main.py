# GitHub #10248: a tuple subscript in a list literal raises in element order.
def f(i: int, z: int) -> int:
    t = (1, 2, 3)
    try:
        xs = [1 // z, t[i]]
    except ZeroDivisionError:
        return 1
    except IndexError:
        return 2
    return xs[1]


assert f(3, 0) == 1
assert f(3, 1) == 2
assert f(1, 1) == 2
