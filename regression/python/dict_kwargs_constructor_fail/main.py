def main() -> None:
    # dict(a=1)["a"] is 1, not 2.
    d = dict(a=1)
    assert d["a"] == 2


main()
