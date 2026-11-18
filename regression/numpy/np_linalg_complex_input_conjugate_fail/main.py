import numpy as np

# A complex matrix's eigenvalues come in no conjugate pairs: these sum to
# the trace, 6 + 1.5j.
a = np.array([[1.0 + 1.0j, 2.0, 0.0], [0.0, 2.0 + 0.5j, 1.0], [1.0j, 0.0, 3.0]])
w = np.linalg.eigvals(a)
assert abs(w[0].imag + w[1].imag + w[2].imag) < 1e-6
