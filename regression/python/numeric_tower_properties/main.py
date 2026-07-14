def main() -> None:
    # int exposes the full numeric tower: n == n/1, with a zero imaginary part.
    n = 5
    assert n.numerator == 5
    assert n.denominator == 1
    assert n.real == 5
    assert n.imag == 0

    # Holds for negatives too.
    m = -7
    assert m.numerator == -7 and m.denominator == 1

    # float exposes real/imag (a real number is its own real part).
    f = 2.5
    assert f.real == 2.5
    assert f.imag == 0.0


main()
