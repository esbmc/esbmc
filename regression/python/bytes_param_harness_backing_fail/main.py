def f(data: bytes) -> int:
    total = 0
    for b in data:
        total += b
    assert total == 0
    return total
