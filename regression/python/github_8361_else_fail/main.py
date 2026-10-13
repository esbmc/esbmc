def first_square_at_least(n: int, limit: int) -> int:
    k = 0
    while (m := k * k) < n:
        if m > limit:
            break
        k += 1
    else:
        return m
    return -m


assert first_square_at_least(100, 20) == 25
