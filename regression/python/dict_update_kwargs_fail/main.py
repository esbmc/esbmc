def main() -> None:
    # After d.update(b=2), d["b"] is 2, not 9.
    d = {"a": 1}
    d.update(b=2)
    assert d["b"] == 9


main()
