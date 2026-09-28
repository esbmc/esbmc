# The overflowed literal is +inf, not a finite value (#7545).
def main() -> None:
    x = 1e400
    assert x < 0.0


main()
