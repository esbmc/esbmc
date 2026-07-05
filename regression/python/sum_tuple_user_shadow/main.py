# A user-defined sum() that shadows the builtin must be honoured even when
# the argument is a tuple — the builtin tuple-fold must not intercept it.
def sum(t):
    return 42


assert sum((1, 2, 3)) == 42
