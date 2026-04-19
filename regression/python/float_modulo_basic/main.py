def test_basic_float_modulo():
    x: float = 5.5
    y: float = 2.0
    z: float = x % y
    assert z == 1.5

def test_equal_operands():
    x: float = 2.2
    y: float = 2.2
    z: float = x % y
    assert z == 0.0

def test_negative_dividend():
    # Python: -5.5 % 2.0 = 0.5 (result has sign of divisor y=2.0)
    x: float = -5.5
    y: float = 2.0
    z: float = x % y
    assert z == 0.5

def test_negative_divisor():
    # Python: 5.5 % -2.0 = -0.5 (result has sign of divisor y=-2.0)
    x: float = 5.5
    y: float = -2.0
    z: float = x % y
    assert z == -0.5

def test_both_negative():
    # Python: -5.5 % -2.0 = -1.5 (result has sign of divisor y=-2.0)
    x: float = -5.5
    y: float = -2.0
    z: float = x % y
    assert z == -1.5

def test_positive_operands():
    x: float = 7.0
    y: float = 3.0
    z: float = x % y
    assert z == 1.0

test_basic_float_modulo()
test_equal_operands()
test_negative_dividend()
test_negative_divisor()
test_both_negative()
test_positive_operands()
