# [True, False, True].count(True) is 2, not 1 — the assertion must fail.
l = [True, False, True]
assert l.count(True) == 1
