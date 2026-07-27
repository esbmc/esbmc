def main() -> None:
    # float.fromhex() on a constant string folds to the exact double, the
    # inverse of float.hex(): "0x", a leading mantissa digit, hex fraction
    # digits, and a binary exponent "p{sign}{n}".
    assert float.fromhex("0x1.c000000000000p+1") == 3.5
    assert float.fromhex("0x1.0000000000000p+0") == 1.0
    assert float.fromhex("0x1.0000000000000p-1") == 0.5
    assert float.fromhex("0x1.999999999999ap-4") == 0.1
    assert float.fromhex("0x1.9000000000000p+6") == 100.0
    assert float.fromhex("0x1.ff00000000000p+7") == 255.5

    # A short mantissa (no padding to 13 digits) and a missing fraction part.
    assert float.fromhex("0x1.8p3") == 12.0
    assert float.fromhex("0x1p3") == 8.0

    # Sign is carried on the whole value.
    assert float.fromhex("-0x1.c000000000000p+1") == -3.5
    assert float.fromhex("-0x1.0000000000000p-1") == -0.5

    # Surrounding ASCII whitespace is stripped, as CPython does.
    assert float.fromhex("  0x1.8p3  ") == 12.0

    # Very large, very small, and subnormal values round-trip exactly.
    assert float.fromhex("0x1.7e43c8800759cp+996") == 1e300
    assert float.fromhex("0x1.56e1fc2f8f359p-997") == 1e-300
    assert float.fromhex("0x0.0000000000001p-1022") == 5e-324


main()
