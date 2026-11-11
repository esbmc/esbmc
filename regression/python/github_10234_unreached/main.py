BODIES = {"a": ([1.0, 2.0], 3.0), "b": ([4.0, 5.0], 6.0)}
SYSTEM = list(BODIES.values())
s = 0.0
for ([x, y], m) in SYSTEM:
    if m < 0.0:
        s += x * m
assert s == 0.0
