# The reversed list's first element is 3, not 1: a stale index->type mapping
# after reverse() must not make this assertion pass.
l = [1, 2.5, 3]
l.reverse()
assert l[0] == 1
