def main() -> None:
    d = {"a": 1, "b": 2}
    d.clear()

    # The dict is now empty.
    assert len(d) == 0
    assert "a" not in d

    # And it remains usable after clearing.
    d["c"] = 3
    assert d["c"] == 3
    assert len(d) == 1


main()
