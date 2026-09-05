# A float literal that overflows to infinity has no JSON number form, so the
# parser tags it rather than emitting the bare token json.dump would (#7545).
def main() -> None:
    x = 1e400
    assert x > 0.0

    y = -1e400
    assert y < 0.0

    z = 1e400j
    assert z.imag > 0.0

    n = -1e400j
    assert n.imag < 0.0

    # The already-working spelling must agree with the literal.
    assert x == float("inf")
    assert y == float("-inf")


main()
