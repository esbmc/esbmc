# Issue #9776: an import alias hid the call from parameter typing.
import lib as L

n = nondet_int()
b = L.f(n)
assert b >= 0
