def main() -> None:
    # update(a=10) sets d["a"] to 10, not 1.
    d = {"a": 1}
    d.update(a=10)
    assert d["a"] == 1


main()
