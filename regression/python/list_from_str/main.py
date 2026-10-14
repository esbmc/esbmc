def main() -> None:
    # list() over a string yields a list of single-character strings.
    x = list("abc")
    assert x[0] == "a" and x[1] == "b" and x[2] == "c" and len(x) == 3

    # A string held in a variable works too.
    s = "hi"
    y = list(s)
    assert y[0] == "h" and y[1] == "i" and len(y) == 2

    # An empty string yields an empty list.
    e = list("")
    assert len(e) == 0

    # list() over a list / tuple is unaffected.
    z = list([1, 2])
    assert z[0] == 1 and z[1] == 2 and len(z) == 2
    w = list((3, 4))
    assert w[0] == 3 and w[1] == 4 and len(w) == 2


main()
