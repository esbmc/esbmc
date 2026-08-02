def main() -> None:
    f = 2.5
    # float is not a Rational in CPython: it has .real/.imag but no
    # .numerator/.denominator. This must stay a clean AttributeError, not
    # silently reuse the int numerator/denominator fold.
    x = f.numerator
    assert x == 2


main()
