def main() -> None:
    # Negative: default is 42, not 0.
    assert min([], default=42) == 0
main()
