def check_type(val):
    if isinstance(val, int):
        return val * 2
    else:
        return 0

result = check_type(5)
assert result == 10
