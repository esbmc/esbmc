def main() -> None:
    a = [(1, "x"), (2, "y")]
    assert a[0] == (1, "x")

    b = [("x", 1), ("y", 2)]
    assert b[1] == ("y", 2)

    c = [("ab", "cd"), ("ef", "gh")]
    assert c[0] == ("ab", "cd")

    p, q = a[1]
    assert p == 2 and q == "y"


main()
