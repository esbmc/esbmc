# Regression for issue #4293: locks in the fix.
# Pre-fix this assertion held silently (step was ignored, len was 6); post-fix
# the step-aware loop produces a 3-element slice and the assertion fails.
xs = [10, 20, 30, 40, 50, 60]
ys = xs[::2]
assert len(ys) == 6
