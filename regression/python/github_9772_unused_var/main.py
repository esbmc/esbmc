def bench(loops):
    range_it = range(loops)
    n = 0
    for _ in range_it:
        n += 2
    return n


assert bench(3) == 6
