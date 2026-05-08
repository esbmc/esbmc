def main() -> None:
    # Empty iterable returns the default.
    assert min([], default=42) == 42
    assert max([], default=-1) == -1
    # Non-empty iterable ignores the default.
    assert min([3, 1, 2], default=99) == 1
    assert max([3, 1, 2], default=99) == 3
main()
