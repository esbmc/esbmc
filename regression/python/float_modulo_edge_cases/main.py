def test_zero_dividend():
    x: float = 0.0
    y: float = 5.0
    z: float = x % y
    assert z == 0.0

def test_small_numbers():
    x: float = 0.1
    y: float = 0.3
    z: float = x % y
    assert z == 0.1

def test_large_dividend():
    x: float = 1000.5
    y: float = 3.0
    z: float = x % y
    assert z == 1.5

def test_fractional_divisor():
    x: float = 5.0
    y: float = 1.5
    z: float = x % y
    assert z == 0.5

def test_negative_zero_result():
    # When result should be exactly zero with negative divisor
    x: float = 6.0
    y: float = -3.0
    z: float = x % y
    assert z == 0.0

test_zero_dividend()
test_small_numbers()
test_large_dividend()
test_fractional_divisor()
test_negative_zero_result()

