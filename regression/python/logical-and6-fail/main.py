x: int = 1
y: int = 0
if (x == 1 and (5 / y) > 0):  # Should cause division by zero in condition
    result: float = 1 / 0
