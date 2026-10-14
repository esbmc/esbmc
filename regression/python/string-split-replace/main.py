def main() -> None:
    # split() on a str.replace() receiver bound through a variable
    s = "a-b-c".replace("-", ".")
    parts = s.split(".")
    assert len(parts) == 3
    assert parts[0] == "a"
    assert parts[2] == "c"

    # replace() with a count argument, inline receiver
    t = "a.a.a".replace(".", "-", 1).split(".")
    assert len(t) == 2
    assert t[0] == "a-a"


main()
