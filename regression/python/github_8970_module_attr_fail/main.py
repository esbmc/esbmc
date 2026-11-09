# On the `right.pick` path `r` is None, so the assertion fails.
import left as l
import right

h = l.pick if nondet_bool() else right.pick
r = h(41)
assert r == 42
