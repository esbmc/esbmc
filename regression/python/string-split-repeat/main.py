def main() -> None:
    # split() on a "*"-repeat receiver, str * n
    parts = ("a." * 2).split(".")
    assert len(parts) == 3      # ['a', 'a', '']
    assert parts[0] == "a"
    assert parts[2] == ""

    # n * str form
    chunks = (3 * "x,").split(",")
    assert len(chunks) == 4      # ['x', 'x', 'x', '']
    assert chunks[0] == "x"


main()
