s1 = {1, 2}
s2 = {1, 2, 3}
# Set ordering (subset/superset) is not yet modelled; ESBMC must reject it
# explicitly instead of silently returning the (wrong) equality result.
r = s1 < s2
assert r
