import math

# CPython rounds the sum of squares once: 13.285352836865115, where
# hypot(hypot(1.94, 9.93), 8.61) gives 13.285352836865117.
r = math.hypot(1.94, 9.93, 8.61)
assert r != 13.285352836865115
