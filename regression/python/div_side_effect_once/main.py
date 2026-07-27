# The divisor of a division is referenced by both the ZeroDivisionError guard
# and the division itself. It must be evaluated exactly once (Python semantics),
# even as the RHS of an assignment (which the frontend converts more than once).
# Here the divisor is a call with a side effect: it must run once.
calls = 0


def divisor() -> int:
    global calls
    calls = calls + 1
    return 2


def main() -> None:
    x = 10 // divisor()
    assert x == 5
    assert calls == 1


main()
