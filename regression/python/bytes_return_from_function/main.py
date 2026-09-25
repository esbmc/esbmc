# Exercises a function returning `-> bytes`, both assigned to a variable
# and indexed/sliced directly from the call result.


def make_literal() -> bytes:
    return bytes([10, 20, 30, 40, 50])


def make_nondet() -> bytes:
    return nondet_bytes(32)


def main() -> None:
    # Assigned to a variable: length and contents must survive the call.
    h = make_literal()
    assert len(h) == 5
    assert h[0] == 10 and h[4] == 50

    n = make_nondet()
    assert len(n) == 32

    # Used directly inline, with no intermediate variable.
    assert len(make_literal()[1:4]) == 3
    assert make_literal()[0:3][0] == 10
    assert len(make_nondet()[0:8]) == 8


main()
