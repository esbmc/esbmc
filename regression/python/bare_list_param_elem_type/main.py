# A bare `list` annotation names no element type, so a subscript of the
# parameter used to read as Any: arithmetic on it raised a spurious TypeError
# and equality silently evaluated false. The element type is recovered from
# the call sites instead. Issue #7187.


def use(xs: list) -> int:
    return xs[0] + 0


def cmp_(xs: list) -> int:
    if xs[0] == 5:
        return 1
    return 0


def use_typed(xs: list[int]) -> int:
    return xs[0] + 0


def main() -> None:
    xs = [5, 3]
    assert use(xs) == 5
    assert cmp_(xs) == 1
    assert use_typed(xs) == 5
    # A list literal at the call site types the parameter just as a name does.
    assert use([7, 1]) == 7


main()
