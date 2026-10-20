import numpy as np

a = np.array([[1, 2], [3, 4]])
chunks = 0
for chunk in np.nditer(a, flags=["external_loop"]):
    chunks += 1
    assert len(chunk) == 4
    assert chunk[0] == 1 and chunk[3] == 4
    assert chunk.sum() == 10
assert chunks == 1

columns = 0
for chunk in np.nditer(a, flags=["external_loop"], order="F"):
    assert len(chunk) == 2
    assert chunk[1] - chunk[0] == 2
    columns += 1
assert columns == 2
