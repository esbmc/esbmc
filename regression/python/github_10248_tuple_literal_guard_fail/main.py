# GitHub #10248: the exception of a tuple-literal element reaches its handler.
def f(i: int, z: int) -> int:
    l = [1, 2, 3]
    try:
        xs = (1 // z, l[i])
    except ZeroDivisionError:
        return 1
    except IndexError:
        return 2
    return xs[1]


assert f(3, 0) == 2
