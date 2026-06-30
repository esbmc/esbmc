# Set equality / inequality must keep working after the ordering guard:
# these go through the Eq/NotEq path, not the rejected ordering path.
s1 = {1, 2, 3}
s2 = {3, 2, 1}
assert s1 == s2
assert not (s1 != s2)

s3 = {1, 2}
assert s1 != s3
assert not (s1 == s3)
