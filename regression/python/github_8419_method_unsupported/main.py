replace = True


def other(self):
    return 7


def dec(f):
    return other if replace else f


class C:
    @dec
    def g(self):
        return 1


assert C().g() == 7
