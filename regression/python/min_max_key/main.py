def main() -> None:
    # max/min with key=abs over a constant list returns the element whose key
    # is largest/smallest (the element itself, not its key).
    assert max([1, -5, 3], key=abs) == -5
    assert min([1, -5, 3], key=abs) == 1
    assert min([5, -2, 8], key=abs) == -2

    # Ties break toward the first occurrence (CPython semantics).
    assert max([-2, 2, 1], key=abs) == -2

    # key=len over a list of string literals.
    assert max(["a", "bbb", "cc"], key=len) == "bbb"
    assert min(["aa", "b", "ccc"], key=len) == "b"

    # The existing lambda x: x[K] path still selects the right element.
    r = max([(1, 9), (2, 3)], key=lambda t: t[1])
    assert r[0] == 1 and r[1] == 9


main()
