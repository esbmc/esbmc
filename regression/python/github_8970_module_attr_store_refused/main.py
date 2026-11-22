# `right.pick` is rebound to `abs`, so it is not the def in right.py.
import left
import right

right.pick = abs
h = left.pick if nondet_bool() else right.pick
r = h(41)
assert r == 42 or r == 41
