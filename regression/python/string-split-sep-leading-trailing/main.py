def main() -> None:
    s = ",a,"
    parts = s.split(",")
    assert parts[0] == ""
    assert parts[1] == "a"
    assert parts[2] == ""


main()
