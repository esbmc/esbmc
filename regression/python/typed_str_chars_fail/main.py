def count_a(s) -> int:
    n = 0
    for c in s:
        if c == 'a':
            n += 1
    return n


def main() -> None:
    s = nondet_str()
    if len(s) > 3:
        return
    assert count_a(s) < 2


main()
