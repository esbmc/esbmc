xs: list[float] = [1.0, 2.0, 3.0]
result: float = 0.0
xl = [] # empty list

# Test with empty list
if xl:
    result = 1.0
else:
    result = 2.0

assert result == 1.0  # List was empty
