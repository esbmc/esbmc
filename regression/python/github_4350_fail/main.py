def main() -> None:
    # Negative: t[1:3] yields (2, 3), so s[0] is 2, not 99.
    t = (1, 2, 3, 4, 5)
    s = t[1:3]
    assert s[0] == 99
main()
