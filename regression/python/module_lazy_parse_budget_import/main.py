import cmath


def main() -> None:
    z: complex = cmath.sqrt(complex(-1.0, 0.0))
    assert z.imag == 1.0


main()
