# `swap` rebinds `right` to `third`, so `right.pick` is not the def in right.py.
import left
import right
import third


def swap():
    global right
    right = third


swap()
h = left.pick if nondet_bool() else right.pick
r = h(41)
assert r == 42 or r == 40
