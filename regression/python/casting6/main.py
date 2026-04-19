# Case 1: Direct float vs char equality
f = float(65)
assert not (f == 'A')  # float vs str: always False

# Case 2: Negative float vs character
f = float(-65)
assert not (f == 'A')  # False

# Case 3: Float from division vs character
f = 130 / 2  # 65.0
assert not (f == 'A')  # False

# Case 4: Float cast from int comparison
i = 65
f = float(i)
assert not (f != 'A') == False  # Also False

# Case 5: Chained operations
a = 60
b = 5
f = float(a + b)
assert (f == 'A') == False

# Case 6: Variable with different types
val = float(66)
char = 'B'
assert not (val == char)  # False


