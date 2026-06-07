# Unpacking a non-tuple element into a nested pattern is a TypeError in Python
# (cannot unpack non-iterable int). ESBMC must reject it explicitly rather than
# cast a non-struct component, which guards the nested-unpacking recursion.
pair = (5, 3)
(a, b), c = pair
