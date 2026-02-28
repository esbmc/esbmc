import math


def test_sqrt_basic():
    """Test: sqrt(4) = 2.0"""
    result: float = math.sqrt(4)
    assert result == 2.0


def test_sqrt_variable():
    """Test: sqrt of variable"""
    x: int = 16
    result: float = math.sqrt(x)
    assert result == 4.0


def test_sqrt_zero():
    """Test: sqrt(0) = 0.0"""
    result: float = math.sqrt(0)
    assert result == 0.0


def test_sqrt_one():
    """Test: sqrt(1) = 1.0"""
    result: float = math.sqrt(1)
    assert result == 1.0


def test_sqrt_expression():
    """Test: sqrt used in larger expression"""
    a: int = 3
    b: int = 4
    # Pythagorean theorem: c = sqrt(a^2 + b^2)
    c: float = math.sqrt(a * a + b * b)
    assert c == 5.0


def test_sqrt_float():
    """Test: sqrt with float input"""
    x: float = 2.25
    result: float = math.sqrt(x)
    assert result == 1.5


def test_sqrt_conditional():
    """Test: sqrt result used in conditional"""
    x: int = 9
    root: float = math.sqrt(x)
    if root == 3.0:
        print("passed: sqrt(9) == 3.0")
    else:
        assert False, "sqrt(9) should be 3.0"


def test_sqrt_return_type():
    """Test: sqrt always returns float even with perfect squares"""
    result: float = math.sqrt(4)  # Should be 2.0, not 2
    # In Python, sqrt always returns float
    assert isinstance(result, float)


def test_sqrt_assignment():
    """Test: sqrt result assigned to variable"""
    x: int = 25
    y: float = math.sqrt(x)
    z: float = y * 2
    assert z == 10.0  # sqrt(25) * 2 = 5 * 2 = 10


def test_sqrt_large():
    """Test: sqrt of large number"""
    x: int = 1000000
    result: float = math.sqrt(x)
    assert result == 1000.0


def test_sqrt_comparison():
    """Test: Comparing sqrt results"""
    sqrt_4: float = math.sqrt(4)
    sqrt_9: float = math.sqrt(9)
    assert sqrt_4 < sqrt_9


def compute_sqrt(n: int) -> float:
    """Helper function that computes sqrt"""
    return math.sqrt(n)


def test_sqrt_parameter():
    """Test: sqrt with function parameter"""
    result: float = compute_sqrt(64)
    assert result == 8.0


def get_sqrt(value: int) -> float:
    """Function that returns sqrt directly"""
    return math.sqrt(value)


def test_sqrt_return():
    """Test: sqrt in return statement"""
    result: float = get_sqrt(49)
    assert result == 7.0


def main():
    test_sqrt_basic()
    test_sqrt_variable()
    test_sqrt_zero()
    test_sqrt_one()
    test_sqrt_expression()
    test_sqrt_float()
    test_sqrt_conditional()
    test_sqrt_return_type()
    test_sqrt_assignment()
    test_sqrt_large()
    test_sqrt_comparison()
    test_sqrt_parameter()
    test_sqrt_return()


if __name__ == "__main__":
    main()
