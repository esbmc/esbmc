# The `else` clause runs only when the condition fails, seeing the binding
# from that final evaluation; `break` skips it.
def first_square_at_least(n: int, limit: int) -> int:
    k = 0
    while (m := k * k) < n:
        if m > limit:
            break
        k += 1
    else:
        return m
    return -m


assert first_square_at_least(10, 100) == 16
assert first_square_at_least(100, 20) == -25
