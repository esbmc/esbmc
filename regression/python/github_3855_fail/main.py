# float() on integer expression should produce the correct float value;
# asserting an incorrect value must fail.
a = 60
b = 5
result = float(a + b)
assert result == 66.0  # wrong: actual is 65.0
