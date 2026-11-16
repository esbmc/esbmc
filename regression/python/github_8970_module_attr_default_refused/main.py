# `right.pick` has a default, which the lifted function would not keep.
import left
import right

h = left.pick if nondet_bool() else right.pick
r = h(41)
assert r == 42 or r is None
