def main() -> None:
    # round(x, -n) with a negative ndigits literal was not constant-folded (the
    # literal -2 is a UnaryOp, not a Constant), so it fell through to the
    # expensive symbolic path and timed out. It now folds and rounds to the
    # nearest power of ten, ties to even.
    assert round(1234, -2) == 1200
    assert round(1234, -1) == 1230
    assert round(1267, -2) == 1300
    assert round(1250, -2) == 1200   # ties to even (12.5 -> 12)
    assert round(1350, -2) == 1400   # ties to even (13.5 -> 14)
    assert round(-1234, -2) == -1200
    assert round(1234.5, -2) == 1200.0

    # Positive / zero ndigits are unchanged.
    assert round(1234, 2) == 1234
    assert round(3.14159, 2) == 3.14


main()
