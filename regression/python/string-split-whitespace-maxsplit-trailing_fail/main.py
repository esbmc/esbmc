# Negative variant: asserting the old (incorrect) stripped remainder must fail.
def main() -> None:
    a = 'a  b  '.split(None, 1)
    assert a[1] == 'b'  # wrong: CPython keeps trailing ws -> 'b  '


main()
