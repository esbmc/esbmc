def times(k):

    def dec(f):

        def w() -> int:
            return k * f()

        return w

    return dec


def n() -> int:
    return 3


@times(n())
def g() -> int:
    return 2


assert g() == 6
