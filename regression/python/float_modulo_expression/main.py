def test_modulo_in_expression():
    x: float = 10.5
    y: float = 3.0
    result: float = (x % y) + 1.0
    assert result == 2.5

def test_chained_modulo():
    x: float = 17.0
    y: float = 5.0
    z: float = 2.0
    result: float = (x % y) % z
    assert result == 0.0

def test_modulo_with_arithmetic():
    a: float = 7.5
    b: float = 2.0
    c: float = (a % b) * 2.0
    assert c == 3.0

def test_negative_in_expression():
    x: float = -10.0
    y: float = 3.0
    result: float = (x % y) + 5.0
    assert result == 7.0  # 2.0 + 5.0 = 7.0

test_modulo_in_expression()
test_chained_modulo()
test_modulo_with_arithmetic()
test_negative_in_expression()
