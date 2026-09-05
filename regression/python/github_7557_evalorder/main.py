# Keywords written out of the builtin's declaration order must not be moved
# into positional slots: Python evaluates arguments in source order, and
# reordering these two calls crashes the converter (#7557). The property that
# actually fails is an unrelated pre-existing "uncaught exception: TypeError"
# false alarm, not seen[0] == 1; what this pins is that a verdict is reached
# at all, rather than a SIGSEGV.
seen: list[int] = []


def f() -> int:
    seen.append(1)
    return 4


def g() -> int:
    seen.append(2)
    return 2


def main() -> None:
    pow(exp=f(), base=g())
    assert seen[0] == 1


main()
