# The element types recorded for xs are not exact here, so the nested
# unpacking is refused.
def g() -> None:
    xs = [[1, 2]]
    ys = xs
    ys.append(["x", "y"])
    ys[0] = [7, 8]
    [a, b], [p, q] = xs
    assert a == 7 and q == "x"


g()
