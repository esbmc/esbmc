def isqrt_like(n: int) -> int:
    x: int = n
    y: int = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


assert isqrt_like(16) == 4
