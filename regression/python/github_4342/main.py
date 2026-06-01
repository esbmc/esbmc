def main() -> None:
    # Positional start.
    assert sum([1, 2, 3], 10) == 16
    # Keyword start.
    assert sum([4, 5, 6], start=100) == 115
    # Default start (no second arg) still works.
    assert sum([1, 2, 3]) == 6
    # Empty iterable returns the start.
    assert sum([], 7) == 7
main()
