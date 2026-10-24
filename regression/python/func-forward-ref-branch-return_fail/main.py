c = True


def a():
    return b()


def b():
    if c:
        return 2
    return 1


assert a() == 1
