def main() -> None:
    # "abcabc".rindex("b") == 4 (last occurrence), not 1.
    assert "abcabc".rindex("b") == 1


main()
