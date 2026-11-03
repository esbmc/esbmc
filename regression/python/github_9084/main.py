# Issue #9084: str() of a call runs the call and converts its result.
trace = 0


def label(x: int) -> str:
    if x > 0:
        return "pos"
    if x < 0:
        raise ValueError("neg")
    return "zero"


def num(x: int) -> int:
    global trace
    trace = trace * 10 + 2
    return x * 2


def head() -> str:
    global trace
    trace = trace * 10 + 1
    return "n"


def main() -> None:
    x = nondet_int()
    try:
        s = str(label(x))
        assert x >= 0
        assert s == "pos" or s == "zero"
    except ValueError:
        assert x < 0
    assert head() + str(num(21)) == "n42"
    assert trace == 12


main()
