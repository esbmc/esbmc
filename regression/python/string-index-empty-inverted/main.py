def main() -> None:
    s = "abc"
    got_error = False
    try:
        s.index("", 2, 1)
    except ValueError:
        got_error = True

    assert got_error


main()
