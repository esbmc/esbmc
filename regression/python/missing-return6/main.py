def safe_divide(a: int, b: int) -> int:
    if b == 0:
        return 0
    else:
        return a // b


result = safe_divide(10, 2)
assert result == 5

result2 = safe_divide(10, 0)
assert result2 == 0
