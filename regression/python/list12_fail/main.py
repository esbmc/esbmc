def sum_while(n: list[int]) -> int:
    l = len(n)
    i = 0
    s = 0
    while i < l:
        s += n[i]
        i += 1
    return s


assert sum_while([1, 2]) == 2
