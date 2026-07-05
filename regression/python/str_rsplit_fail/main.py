def main() -> None:
    # rsplit counts from the right, so "a.b.c".rsplit(".", 1) is
    # ["a.b", "c"], not ["a", "b.c"] (which is split(".", 1)).
    assert "a.b.c".rsplit(".", 1) == ["a", "b.c"]


main()
