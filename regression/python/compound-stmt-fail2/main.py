def test_div_zero():
    x = 0
    x += 1
    x -= 1
    x /= x  # division by zero


test_div_zero()
