def main() -> None:
    # Numeric min()/max() over multiple direct arguments is unaffected by the
    # non-numeric guard.
    assert max(3, 7, 2) == 7
    assert min(5, 3, 9) == 3
    assert max(3.5, 2) == 3.5
    assert max(True, False) is True

    # The single-iterable form over strings still works (lexicographic, via the
    # model) — this is the supported way to take a max/min over strings.
    assert max(["apple", "banana"]) == "banana"
    assert min(["apple", "banana"]) == "apple"


main()
