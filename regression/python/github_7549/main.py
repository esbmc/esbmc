# A class or a builtin type name bound to a value is a class object: usable in
# a class body, as a default argument, and distinguishable from another class
# (#7549).
class C:
    pass


class D:
    pass


class MyErr(Exception):
    pass


class Holder:
    t = int
    u = C


def f(x: int, exc=MyErr) -> int:
    return x


def main() -> None:
    x = C
    y = C
    assert x == y
    assert x != D
    assert f(1) == 1


main()
