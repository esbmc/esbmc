def main() -> None:
    # bool is an int subclass; abs() of a bool yields an int (abs(True) == 1,
    # abs(False) == 0). Previously abs() built a `bool >= 0` comparison whose
    # mismatched sorts tripped a solver assertion (a crash, not a verdict).
    assert abs(True) == 1
    assert abs(False) == 0

    # A bool variable and a boolean expression both work.
    b = True
    assert abs(b) == 1
    assert abs(2 > 1) == 1
    assert abs(1 > 2) == 0

    # The result is an int and composes in arithmetic.
    assert abs(True) + 10 == 11

    # abs() on int / float is unchanged.
    assert abs(-5) == 5
    assert abs(-3.5) == 3.5


main()
