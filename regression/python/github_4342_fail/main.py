def main() -> None:
    # Negative: with start=10, the sum is 16, not 6.
    assert sum([1, 2, 3], 10) == 6
main()
