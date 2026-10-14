def main() -> None:
    # tuple() over a string yields a tuple of single-character strings.
    t = tuple("ab")
    assert t[0] == "a" and t[1] == "b" and len(t) == 2

    # A string held in a variable works too.
    s = "abc"
    u = tuple(s)
    assert u[0] == "a" and u[2] == "c" and len(u) == 3

    # An empty string yields an empty tuple.
    e = tuple("")
    assert len(e) == 0

    # tuple() over a list / tuple is unaffected.
    v = tuple([1, 2])
    assert v[0] == 1 and v[1] == 2


main()
