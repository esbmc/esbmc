import numpy as np

# A chain deeper than the supported bound must decline explicitly instead of
# crashing or silently producing a wrong verdict.
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
assert r[0] == False
