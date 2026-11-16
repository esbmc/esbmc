# Issue #9776: an import alias hid the call from parameter typing.
import lib as L

b = L.inc(2.5)
assert b == 3.5
assert b > 4.0
