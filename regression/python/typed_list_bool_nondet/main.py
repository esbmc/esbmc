def main() -> None:
    perm = nondet_list(nondet_int)
    n = len(perm)
    if n > 4:
        return
    seen = [False] * n
    for x in perm:
        if not (0 <= x < n) or seen[x]:
            return
        seen[x] = True
    for i in range(n):
        assert seen[i]


main()
