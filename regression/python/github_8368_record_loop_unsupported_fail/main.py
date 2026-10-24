# The element types recorded for xs are not exact here, so the nested
# unpacking is refused.
xs = [[1, 2]]
for i in range(1):
    xs.append(["x", "y"])
[a, b], [p, q] = xs
assert a == 1 and p == "y" and q == "y"
