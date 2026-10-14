import numpy as np

# A chain deeper than the supported bound must decline explicitly instead of
# crashing or silently producing a wrong verdict. This must never reach the
# assert below -- but if the supported chaining depth is ever raised and the
# rejection no longer fires here, the six logical_not() calls cancel out in
# pairs (an even count), so real NumPy leaves equal(a, b)[0] unchanged (True),
# not negated.
a = [1, 2, 3]
b = [1, 5, 3]
r = np.logical_not(
    np.logical_not(
        np.logical_not(
            np.logical_not(
                np.logical_not(np.logical_not(np.equal(a, b)))
            )
        )
    )
)
assert r[0] == True
