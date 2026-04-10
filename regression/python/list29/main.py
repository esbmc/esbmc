xs: list[float] = [1.0, 2.0, 3.0]
result: float = 0.0

# Test with non-empty list
if xs:
    result = 1.0
else:
    result = 2.0

assert result == 1.0  # List was non-empty

# Pop all elements
xs.pop()
xs.pop()
xs.pop()

# Test with empty list
if xs:
    result = 3.0
else:
    result = 4.0

assert result == 4.0  # List was empty
