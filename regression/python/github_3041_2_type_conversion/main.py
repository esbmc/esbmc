def test_float_to_int(x: float) -> bool:
    return int(x) == 3


def test_bool_true(x: bool) -> bool:
    return int(x) == 1


def test_bool_false(x: bool) -> bool:
    return int(x) == 0


def test_int_to_int(x: int) -> bool:
    return int(x) == 42


assert test_float_to_int(3.14)
assert test_float_to_int(3.99)
assert test_bool_true(True)
assert test_bool_false(False)
assert test_int_to_int(42)
