# Nested generators: all(expr for x in xs for y in ys)
xs = [1, 2]
ys = [10, 20]
assert all(x + y > 0 for x in xs for y in ys) == True

# Nested generators with a falsy result
xs2 = [1, -100]
ys2 = [10, 20]
assert all(x + y > 0 for x in xs2 for y in ys2) == False

# Nested generators with if clause
xs3 = [1, 2, 3]
ys3 = [4, 5, 6]
assert all(x * y > 4 for x in xs3 for y in ys3 if y > 4) == True
