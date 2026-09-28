def shadowed_list():
    c = [0]

    def inner():
        c = [9]
        return c[0]

    v = inner()
    return c[0] + v


def shadowed_scalar():
    n = 1

    def inner():
        n = 9
        return n

    v = inner()
    return n + v


def main():
    assert shadowed_list() == 9
    assert shadowed_scalar() == 10


main()
