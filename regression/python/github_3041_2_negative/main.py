def test_negative(x: str) -> bool:
    return int(x) == -42

def test_positive_sign(x: str) -> bool:
    return int(x) == 42

assert test_negative("-42")
assert test_positive_sign("+42")
