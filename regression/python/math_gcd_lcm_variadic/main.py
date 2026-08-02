import math


def main() -> None:
    # math.gcd / math.lcm accept any number of integer arguments (CPython 3.9+),
    # but the operational model is binary. The preprocessor normalises the
    # variadic forms to nested 2-argument calls.
    assert math.gcd(12, 8, 6) == 2
    assert math.gcd(24, 36, 48, 60) == 12
    assert math.lcm(4, 6, 8) == 24

    # Zero arguments return the identity (gcd -> 0, lcm -> 1).
    assert math.gcd() == 0
    assert math.lcm() == 1

    # One argument returns the absolute value.
    assert math.gcd(12) == 12
    assert math.gcd(-15) == 15
    assert math.lcm(7) == 7

    # The two-argument form is unchanged, and negatives fold correctly.
    assert math.gcd(12, 8) == 4
    assert math.gcd(-12, 8, 6) == 2

    # The variadic fold works on symbolic arguments too (the binary model runs).
    x = 24
    assert math.gcd(x, 12, 8) == 4


main()
