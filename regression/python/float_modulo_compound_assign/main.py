def test_compound_modulo_assign():
    x: float = 10.5
    x %= 3.0
    assert x == 1.5

def test_compound_modulo_negative():
    x: float = -10.5
    x %= 3.0
    assert x == 1.5  # Result has sign of divisor (3.0)

def test_compound_modulo_negative_divisor():
    x: float = 10.5
    x %= -3.0
    assert x == -1.5  # Result has sign of divisor (-3.0)

def test_multiple_compound_assigns():
    x: float = 20.0
    x %= 7.0
    x %= 3.0
    assert x == 0.0

test_compound_modulo_assign()
test_compound_modulo_negative()
test_compound_modulo_negative_divisor()
test_multiple_compound_assigns()
