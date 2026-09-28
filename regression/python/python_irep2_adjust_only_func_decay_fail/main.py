# Negative counterpart: the same decayed function pointer under the hop-off
# path, with a false assertion. The decay must reach the solver and report the
# violation, not mask it behind an unresolvable indirect call.
def get_tripler():
    def triple(x: int) -> int:
        return x * 3

    return triple


f = get_tripler()
assert f(4) == 99
