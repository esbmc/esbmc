# github #9629: x may be a B, whose str() is B.__str__, not A.__repr__.
class A:
    def __repr__(self):
        return "<A>"


class B(A):
    def __str__(self):
        return "B"


def f(x: A) -> str:
    return str(x)


assert f(B()) == "<A>"
