# The class-object fallback for a builtin exception name must sit behind
# symbol lookup, so a name rebound to a value still resolves to that value.
ValueError = 5
assert ValueError == 5
