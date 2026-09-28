# A module that defines its own `nondet_list` must have *its* function called.
# The rewrite to ESBMC's typed builder matches on the callee name, so without a
# shadowing check the user's body is replaced by a model returning a list of
# non-deterministic length, and this exact length is no longer provable.
def nondet_list(n: int) -> list:
    return []

x = nondet_list(3)
assert len(x) == 0
