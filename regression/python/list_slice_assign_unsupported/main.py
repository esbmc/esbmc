# List slice assignment is not modelled. ESBMC must reject it with an explicit
# error rather than silently ignore the store (which previously left the list
# unchanged and reported buggy programs as SUCCESSFUL). This program is valid
# CPython (a becomes [1, 9, 4]).
a = [1, 2, 3, 4]
a[1:3] = [9]
assert a == [1, 9, 4]
