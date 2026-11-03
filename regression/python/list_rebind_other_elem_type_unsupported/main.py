# xs holds ints on one path and strs on the other, which the per-name
# element type record cannot express.
c = nondet_bool()
xs = [1, 2]
if c:
    xs = ["a", "b"]
if not c:
    assert xs[0] == 1
