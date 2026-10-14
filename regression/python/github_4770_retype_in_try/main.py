# Soundness boundary for straight-line retyping (#4770/#4774): a retype that
# occurs inside a conditionally-executed body (here a try block, on a path that
# is skipped by the preceding raise) must NOT be treated as an unconditional
# top-level rebinding. The fix only renames at block nesting depth 1, so this
# retype is left to the existing fallback and the post-try read of x still sees
# the original value 100.
x = 100
assert x == 100
try:
    raise ValueError()
    x = 'hello'          # never executes; must not redirect the read below
except ValueError:
    pass
assert x == 100
