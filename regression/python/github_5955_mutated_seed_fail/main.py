# The assert folder seeded l with its literal [1, 2]: the append is not an
# assignment statement, so the write-once heuristic missed it and the stale
# seed folded this failing assert to True (GitHub #5955).
l = [1, 2]
l.append(3)
assert l.count(3) == 0
