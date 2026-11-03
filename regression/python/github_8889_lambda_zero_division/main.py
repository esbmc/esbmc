# Issue #8889: a division by zero in a lambda body raises ZeroDivisionError.
f = lambda x: 10 // x
try:
    f(0)
    assert False
except ZeroDivisionError:
    pass

g = lambda x: 1 if x == 0 else 10 % x
assert g(0) == 1
assert g(3) == 1
