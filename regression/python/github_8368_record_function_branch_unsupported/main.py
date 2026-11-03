# The element types recorded for xs are not exact here, so the nested
# unpacking is refused.
def g(c: bool) -> None:
    xs = [[1, 2], [3, 4]]
    if c:
        xs = [["a", "b"], ["c", "d"]]
    [a, b], [p, q] = xs
    if c:
        assert a == "a" and q == "d"
    else:
        assert a + q == 5


g(nondet_bool())
