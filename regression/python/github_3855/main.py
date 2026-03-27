# Issue #3855: float() on integer expression must produce the correct float value.
a = 60
b = 5
result = float(a + b)
assert result == 65.0
