# Test nondet_str() with nested conditionals

x = nondet_str()

if x == "a":
    result = 1
elif x == "b":
    result = 2
elif x == "c":
    result = 3
else:
    result = 0

# Result should be one of 0, 1, 2, or 3
assert result >= 0 and result <= 3

