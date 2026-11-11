# GitHub #10248: an element of a tuple literal raises a catchable exception.
def f(i: int, z: int) -> int:
    l = [1, 2, 3]
    try:
        xs = (1 // z, l[i])
    except ZeroDivisionError:
        return 1
    except IndexError:
        return 2
    return xs[1]


assert f(3, 0) == 1
assert f(3, 1) == 2
assert f(1, 1) == 2
