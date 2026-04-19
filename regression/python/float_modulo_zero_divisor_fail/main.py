def test_float_modulo_zero_divisor():
    x: float = 5.0
    y: float = 0.0
    z: float = x % y  # Should raise ZeroDivisionError in Python
    assert False, "Modulo by zero should not succeed"

test_float_modulo_zero_divisor()

