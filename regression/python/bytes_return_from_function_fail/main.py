def make() -> bytes:
    return nondet_bytes(32)


def main() -> None:
    # len(make()[0:8]) is 8, not 999.
    assert len(make()[0:8]) == 999


main()
