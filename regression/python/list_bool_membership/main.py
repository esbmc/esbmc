# Bool membership, count and index over a list (variable receiver, not a
# constant-folded literal) now resolve correctly.
l = [True, False, True]
assert True in l
assert False in l
assert l.count(True) == 2
assert l.count(False) == 1
assert l.index(True) == 0
assert l.index(False) == 1
