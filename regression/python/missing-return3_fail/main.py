def process_number(x: int) -> int:
    if x == 0:
        return 0
    if x < 0:
        return -x
    # Missing return for positive numbers
    x * 2

result = process_number(5)
assert result == 10
