def countdown(n: int) -> list[int]:
    out = []
    i = n
    while i > 0:
        i -= 1
        out.append(i)
    return out


def main() -> None:
    n = nondet_int()
    if n < 0 or n > 3:
        return
    c = countdown(n)
    assert countdown(n) == sorted(c)


main()
