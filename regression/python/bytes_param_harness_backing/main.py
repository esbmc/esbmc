def f(data: bytes) -> int:
    total = 0
    for b in data:
        total += b
    return total
