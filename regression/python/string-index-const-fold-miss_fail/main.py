s = "hello"

# index() raises ValueError when absent, so the fold must decline the miss
# and leave the model to report it.
i = s.index("zz")
assert i >= 0
