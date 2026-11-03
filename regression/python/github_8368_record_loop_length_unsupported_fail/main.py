# The element types recorded for xs are not exact here, so the nested
# unpacking is refused.
xs = [[1, 2]]
for i in range(1):
    xs.append([5, 6, 7])
[a, b], [p, q] = xs
