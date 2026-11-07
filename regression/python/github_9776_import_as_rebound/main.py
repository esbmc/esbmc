# Issue #9776: an alias bound twice must not resolve to main's own g.
import lib
import lib as L
import lib as L


def g(n):
    return n


lib.g(1)
b = L.g("ab")
assert b == "ab"
