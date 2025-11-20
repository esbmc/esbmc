def test_base36(x: str) -> bool:
    # 'Z' in base 36 = 35
    return int(x, 36) == 35

def test_base36_multi(x: str) -> bool:
    # '10' in base 36 = 36
    return int(x, 36) == 36

assert test_base36("Z")
assert test_base36_multi("10")
