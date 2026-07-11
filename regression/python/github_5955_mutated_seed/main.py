# Post-fix contract: a mutated global list is never seeded, so these asserts
# go through the list operational model and verify against the real value.
l = [1, 2]
l.append(3)
assert l.count(3) == 1
assert l == [1, 2, 3]
