# The element types recorded for xs are not exact here, so the nested
# unpacking is refused.
def g() -> None:
    xs = [[1, 2], [3, 4]]
    ys = xs
    ys[1] = [3]
    [a, b], [p, q] = xs


g()
