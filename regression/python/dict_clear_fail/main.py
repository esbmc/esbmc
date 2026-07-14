def main() -> None:
    d = {"a": 1}
    d.clear()
    # The dict is empty after clear(), so len is 0, not 1.
    assert len(d) == 1


main()
