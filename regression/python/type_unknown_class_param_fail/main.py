# x: A may hold an instance of a subclass, so type(x) is A is not known
# statically.
class A:

    def __init__(self) -> None:
        self.x = 1


def f(x: A) -> None:
    assert type(x) is not A


f(A())
