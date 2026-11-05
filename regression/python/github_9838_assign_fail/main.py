# Issue #9838: a missing attribute as an `or` operand of an assignment raises.
class C:
    pass


def f(c):
    x = False or c.limit
    return x


f(C())
