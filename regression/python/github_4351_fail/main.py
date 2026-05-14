def main() -> None:
    # Negative: 99 is not in the comprehension's range.
    s = {i for i in range(3)}
    assert 99 in s
main()
