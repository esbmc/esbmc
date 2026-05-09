def main() -> None:
    pairs = [(1, "b"), (3, "a"), (2, "c")]
    # min by second tuple element.
    assert min(pairs, key=lambda p: p[1])[1] == "a"
    # max by first tuple element.
    assert max(pairs, key=lambda p: p[0])[0] == 3
    # When several elements share the minimum key, the first wins (CPython).
    eq = [(0, "x"), (0, "y")]
    assert min(eq, key=lambda p: p[0])[1] == "x"
main()
