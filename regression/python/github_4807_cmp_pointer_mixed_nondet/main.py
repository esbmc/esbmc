# Regression for #4807: the pointer-vs-non-pointer arm of the comparison
# guard used to abort when one operand was inferred as pointer-backed
# (e.g. a variable rebound from int to list in the same scope) and the
# other wasn't. The handler now returns a sound nondet bool fallback so
# GOTO conversion proceeds.
#
# Minimal pattern: a parameter `n` is compared with `0` and then later
# reassigned to a list -- the frontend infers a list (pointer) type at
# the comparison site. Previously this aborted with "pointer-backed and
# non-pointer". This test only checks that conversion completes; the
# nondet result is unconstrained by design.


def digits_sum(n):
    if n < 0:                                # was abort -- now nondet bool
        n, _neg = -1 * n, -1
    n = [int(i) for i in "12"]
    n[0] = n[0]
    return sum(n)


if __name__ == "__main__":
    # Never actually call digits_sum -- the test exercises the conversion
    # of its body. Absence of an abort during conversion is the assertion.
    pass
