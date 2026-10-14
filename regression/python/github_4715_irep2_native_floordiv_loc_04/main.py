def half_floor(n: int, d: int) -> int:
    return (n + d // n) // 2


assert half_floor(8, 4) == 4
