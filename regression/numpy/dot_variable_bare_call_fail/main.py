import numpy as np

# np.dot(a, b) used as a bare expression statement (its result is never
# assigned) with both operands read from a variable, not a literal. This
# used to abort the process (SIGSEGV during GOTO conversion) instead of
# producing a diagnostic. The runtime dot backend always needs somewhere to
# write its result, so a discarded result with no assignment target is
# explicitly unsupported rather than silently synthesizing storage for it.
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b)
