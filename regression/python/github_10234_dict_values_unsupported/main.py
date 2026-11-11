# The inner lists lose their element types through dict.values(), so x * m
# met a void* and a float and aborted in the solver.
BODIES = {"a": ([1.0, 2.0], 3.0), "b": ([4.0, 5.0], 6.0)}
SYSTEM = list(BODIES.values())
s = 0.0
for ([x, y], m) in SYSTEM:
    s += x * m
assert s == 27.0
