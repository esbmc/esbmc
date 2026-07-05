def main() -> None:
    # Before the fix the arithmetic key was silently dropped, leaving the list
    # in natural order, so this WRONG assertion verified as SUCCESSFUL. It must
    # now be falsified.
    assert sorted([1, 2, 3], key=lambda v: -v) == [1, 2, 3]


main()
