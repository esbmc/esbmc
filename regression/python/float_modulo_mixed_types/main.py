def test_int_mod_float():
    x: int = 5
    y: float = 2.0
    z: float = x % y
    assert z == 1.0

def test_float_mod_int():
    x: float = 5.5
    y: int = 2
    z: float = x % y
    assert z == 1.5

def test_negative_int_mod_float():
    x: int = -7
    y: float = 3.0
    z: float = x % y
    assert z == 2.0  # Python: -7 % 3 = 2, not -1

def test_float_mod_negative_int():
    x: float = 7.5
    y: int = -3
    z: float = x % y
    assert z == -1.5  # Python: 7.5 % -3 = -1.5

test_int_mod_float()
test_float_mod_int()
test_negative_int_mod_float()
test_float_mod_negative_int()
