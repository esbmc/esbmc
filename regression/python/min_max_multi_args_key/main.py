def pick_longer(a: str, b: str) -> str:
    if a == b:
        return a
    # max() over multiple non-numeric arguments with a key= keyword is
    # legitimate (the comparison is on key(x), not the strings themselves).
    # The non-numeric guard must NOT reject this at conversion time, otherwise
    # the whole function fails to convert even when this branch is unreachable
    # for the concrete arguments below.
    return max(a, b, key=len)


def main() -> None:
    # For equal inputs the max(..., key=len) branch is never taken, so the
    # result is well-defined; the test pins that conversion succeeds.
    assert pick_longer("x", "x") == "x"


main()
