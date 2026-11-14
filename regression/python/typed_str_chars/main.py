def count_a(s) -> int:
    n = 0
    for c in s:
        if c == 'a':
            n += 1
    return n


def main() -> None:
    s = nondet_str()
    t = nondet_str()
    if len(s) > 3 or len(t) > 3:
        return
    for c in s:
        if c != 'a' and c != 'b':
            return
    if len(s) == len(t) and len(s) > 0 and s[0] == t[0] and s[0] == 'a':
        assert t[0] != 'b'
    assert count_a(s) <= len(s)
    if len(s) == 2 and s[0] == s[1] and s[1] == 'b':
        assert count_a(s) == 0


main()
