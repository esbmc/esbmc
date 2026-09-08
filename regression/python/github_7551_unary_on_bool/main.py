# bool is a subclass of int, so -True is -1 and ~True is -2. CPython 3.12 warns
# that ~ on a bool is deprecated for removal in 3.16; the arithmetic it performs
# is still the underlying int's (#7551).
def main() -> None:
    assert -True == -1
    assert -False == 0
    assert ~True == -2
    assert ~False == -1

    x: bool = True
    assert -x == -1

    b: bool = False
    assert ~b == -1

    assert -(1 == 1) == -1
    assert +True == 1


main()
