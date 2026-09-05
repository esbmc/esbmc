# A keyword names a builtin's parameter; the value must reach that parameter
# rather than falling back to its default (#7557).
def main() -> None:
    assert int("10", base=2) == 2
    assert str(object=500) == "500"
    assert round(1.567, ndigits=2) == 1.57
    assert round(number=1.5) == 2
    assert pow(base=2, exp=4) == 16
    assert pow(2, 4, mod=3) == 1

    b: int = 2
    assert int("11", base=b) == 3


main()
