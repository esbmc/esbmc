def main() -> None:
    # float.hex() on a constant literal receiver folds to CPython's exact
    # hexadecimal string: "0x" then a leading mantissa digit, 13 hex mantissa
    # digits (the full 52-bit significand), and a binary exponent "p{sign}{n}".
    assert (3.5).hex() == "0x1.c000000000000p+1"
    assert (1.0).hex() == "0x1.0000000000000p+0"
    assert (0.5).hex() == "0x1.0000000000000p-1"
    assert (2.0).hex() == "0x1.0000000000000p+1"
    assert (0.1).hex() == "0x1.999999999999ap-4"
    assert (100.0).hex() == "0x1.9000000000000p+6"
    assert (255.5).hex() == "0x1.ff00000000000p+7"

    # Sign is carried; -0.0 keeps its sign.
    assert (-3.5).hex() == "-0x1.c000000000000p+1"
    assert (-0.5).hex() == "-0x1.0000000000000p-1"

    # Zero collapses to a single "0" mantissa (not 13 padded zeros).
    assert (0.0).hex() == "0x0.0p+0"
    assert (-0.0).hex() == "-0x0.0p+0"

    # Very large, very small, and subnormal values.
    assert (1e300).hex() == "0x1.7e43c8800759cp+996"
    assert (1e-300).hex() == "0x1.56e1fc2f8f359p-997"
    assert (5e-324).hex() == "0x0.0000000000001p-1022"


main()
