import numpy as np

w = np.linalg.eigvals(np.array([[0.0, -1.0], [1.0, 0.0]]))
assert abs(w[0].real) < 1e-9
assert abs(w[0].imag - 1.0) < 1e-9
assert abs(w[1].imag + 1.0) < 1e-9

s = np.linalg.svd(np.array([[1.0, 2.0], [3.0, 4.0]]), compute_uv=False)
assert abs(s[0] - 5.4649857042190426) < 1e-9
assert abs(s[1] - 0.36596619062625746) < 1e-9
assert s[0] >= s[1]
