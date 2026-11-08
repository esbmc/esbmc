def add(cls):
    cls.k = 5
    n = cls.k
    return cls


@add
class C:
    k = 0


assert C.k == 5
