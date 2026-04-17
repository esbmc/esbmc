# Nested generators where not all combinations satisfy the condition
xs = [1, 2]
ys = [10, 20]
assert all(x * y > 15 for x in xs for y in ys) == True
