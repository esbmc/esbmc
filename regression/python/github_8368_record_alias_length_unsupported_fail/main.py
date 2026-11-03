# The element types recorded for xs are not exact here, so the nested
# unpacking is refused.
xs = [[1, 2], [3, 4]]
ys = xs
ys[0] = [3, 4, 5]
[a, b], [p, q] = xs
