# PEP 515: the int()/float() constructors accept a single underscore between
# digits, and after a base specifier; never leading, trailing or doubled.
def main() -> None:
    assert int("1_00") == 100
    assert int("0_100") == 100
    assert int("1_0_0") == 100
    assert int("1_00", 3) == 9
    assert int("0x_1f", 16) == 31
    assert float("1_0.5") == 10.5
    assert float("1_0e1_0") == 100000000000.0

    # The same strings held in a variable fold through a separate path.
    i = "1_00"
    assert int(i) == 100
    f = "1_0.5"
    assert float(f) == 10.5

    # Unchanged forms.
    assert int("100") == 100
    assert int("  100  ") == 100
    assert int("+100") == 100


main()
