# A keys view is set-like, so ordering it against a set is subset comparison,
# which is not modelled. ESBMC must reject it explicitly rather than fall
# through to the list comparator and return a wrong verdict (#7553).
d = {1: 1}
r = d.keys() <= {1, 2}
assert r
