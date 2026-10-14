# A dict comprehension iterating d.items() with a (key, value) tuple target --
# the most common dict-comprehension idiom. The converter's handler models a
# list-of-tuples iterable but not a dict.items() view, so this aborted with
# "Only simple targets are supported in DictComp" / "requires iterating a list
# of tuples". It is now lowered to a dict-building for-loop over items().

d = {1: 10, 2: 20, 3: 30}

# Transform values; both names usable.
e = {k: v + 1 for k, v in d.items()}
assert e[1] == 11
assert e[3] == 31

# Swap key/value.
f = {v: k for k, v in d.items()}
assert f[20] == 2

# With a filter.
g = {k: v for k, v in d.items() if v > 15}
assert g[2] == 20
assert g[3] == 30
