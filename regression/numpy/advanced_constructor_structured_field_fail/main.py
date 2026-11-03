import numpy as np

a = np.array([(1, 2.0)], dtype={"names": ["a", "b"], "formats": ["i4", "f8"]})
assert a[0]["b"] == 2.0
