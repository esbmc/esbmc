# A dict of ints only: a value read with a variable key is an int.
import random

d = {"a": 1, "b": 3}
k = "a" if random.random() < 0.5 else "b"
v = d[k]
assert v == 1 or v == 3
