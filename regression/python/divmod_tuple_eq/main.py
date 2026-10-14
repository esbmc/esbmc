def main() -> None:
    # divmod() returns a tuple. Storing it in a variable and comparing to a
    # tuple literal previously crashed: the divmod result used a hard-coded
    # "tag-tuple_divmod" struct tag, mismatching the content-based tag of the
    # literal, which tripped a solver tuple-sort assertion. The result now uses
    # the shared tuple struct type, so the sorts match.
    x = divmod(17, 5)
    assert x == (3, 2)

    # Negative and float divmod, and the unchanged index / unpack / inline forms.
    assert divmod(-7, 3) == (-3, 2)
    f = divmod(7.5, 2.0)
    assert f == (3.0, 1.5)
    assert x[0] == 3 and x[1] == 2
    q, r = divmod(17, 5)
    assert q == 3 and r == 2


main()
