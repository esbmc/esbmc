# The element types recorded for xs are not exact here, so the nested
# unpacking is refused.
c = nondet_bool()
xs = [[1, 2], [3, 4]]
if c:
    xs = [["a", "b"], ["c", "d"]]
[a, b], [p, q] = xs
if c:
    assert a == "b"
else:
    assert a + q == 5
