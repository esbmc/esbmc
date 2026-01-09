def main() -> None:
    s = "abc"
    try:
        s.index("", 2, 1)
        assert False, "Expected ValueError for empty substring with inverted range"
    except ValueError:
        pass

main()
