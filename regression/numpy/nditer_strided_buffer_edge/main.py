import numpy as np

a = np.array([[0, 1, 2], [3, 4, 5]])
v = a[:, ::2]
chunks = 0
for chunk in np.nditer(v, flags=["external_loop"]):
    assert len(chunk) == 2
    chunks += 1
assert chunks == 2

buffered = 0
for chunk in np.nditer(v, flags=["external_loop", "buffered"]):
    assert len(chunk) == 4
    assert chunk[0] == 0 and chunk[1] == 2 and chunk[2] == 3 and chunk[3] == 5
    buffered += 1
assert buffered == 1
