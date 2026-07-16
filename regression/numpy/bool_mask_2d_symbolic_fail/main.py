import numpy as np

# ADR-NP-001: a symbolic 2-D row mask stays explicitly rejected until the
# canonical ndarray descriptor exists to carry the logical row count as
# part of the value itself - a bounded-capacity result with a detached
# `count` variable would leave len()/shape() observing the physical
# capacity instead of the logical size (see numpy-architecture-decisions.md).
a = np.array([[1, 2], [3, 4]])
n = nondet_bool()
mask = np.array([n, not n])
x = a[mask]
