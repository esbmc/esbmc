# Issue #3855: float() on integer expression causes z3_conv.cpp assertion
# when combined with chr(int(...)) and an equality assertion.
a = 60
b = 5
sum = float(a + b)
assert sum == 65.0
sum = chr(int(sum))
