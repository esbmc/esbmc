def main() -> None:
    # split() on a transitive chain of "+" concatenations bound through a variable
    s = "a." + "b"
    t = s + ".c"
    parts = t.split(".")
    assert len(parts) == 3
    assert parts[0] == "a"
    assert parts[1] == "b"
    assert parts[2] == "c"


main()
