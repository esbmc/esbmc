
def main() -> None:
    l = nondet_list(nondet_int)
    if len(l) > 4:
        return
    i = nondet_int()
    if not 0 <= i <= len(l):
        return
    l[i:] = reversed(l[i:])
    assert len(l) < 4 or l[3] == l[i]


main()
