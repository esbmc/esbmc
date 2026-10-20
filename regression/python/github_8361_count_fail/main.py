def count(n: int) -> int:
    k = 0
    while (m := n - k) > 0:
        k += 1
    return k


assert count(3) == 2
