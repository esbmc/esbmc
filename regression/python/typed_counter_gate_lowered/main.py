from collections import Counter


def main() -> None:
    s = nondet_str()
    c = Counter()
    if len(s) > 0 and s[0] == 'a':
        c[1, 2] = 3
    assert (max(c.values()) if c else 0) == c[1, 2]


main()
