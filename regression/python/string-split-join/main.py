def main() -> None:
    # split() on a str.join() receiver built from an inline list
    parts = ",".join(["a", "b", "c"]).split(",")
    assert len(parts) == 3
    assert parts[0] == "a"
    assert parts[2] == "c"

    # join() over a variable-bound list
    items = ["x", "y"]
    s = "-".join(items).split("-")
    assert len(s) == 2
    assert s[1] == "y"


main()
