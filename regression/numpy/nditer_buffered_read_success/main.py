import numpy as np

a = np.array([[1, 2], [3, 4]])
order = 0
for x in np.nditer(np.transpose(a), flags=["buffered"]):
    order = order * 10 + x
assert order == 1234

chunks = 0
for chunk in np.nditer(a, flags=["buffered", "external_loop"]):
    chunks += 1
    assert len(chunk) == 4
assert chunks == 1
