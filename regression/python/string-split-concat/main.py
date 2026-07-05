def main() -> None:
    # split() on an inline "+" concatenation
    parts = ("a." + "b").split(".")
    assert len(parts) == 2
    assert parts[0] == "a"
    assert parts[1] == "b"

    # split() on a variable bound to a nested concatenation
    s = "x." + "y." + "z"
    chunks = s.split(".")
    assert len(chunks) == 3
    assert chunks[0] == "x"
    assert chunks[1] == "y"
    assert chunks[2] == "z"


main()
