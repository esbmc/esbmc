import numpy as np

# Without the dtype=int truncation the array holds 5.7 and searching for the
# untruncated 5.7 would insert at index 0 (left of the matching elements);
# truncated to 5, 5.7 sorts strictly after every element instead.
i = np.searchsorted(np.full((3,), 5.7, dtype=int), 5.7)

assert i == 3
