import numpy as np

a = np.array([7])
count = 0
for chunk in np.nditer(a, flags=["external_loop"]):
    count += 1
    assert len(chunk) == 1
    assert chunk[0] == 7
assert count == 1

caught = False
try:
    np.nditer(a, flags=["external_loop", "multi_index"])
except ValueError:
    caught = True
assert caught
