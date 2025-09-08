def good_function(x: int) -> int:
    if x > 0:
        return x
    else:
        return 0

def bad_function(y: int) -> int:
    if y > 10:
        return y * 2
    # Missing return for y <= 10

result1 = good_function(5)
result2 = bad_function(5)  # This will trigger the missing return
assert result1 == 5
