# Non-scalar elements give no single copy length, so slicing keeps the model's
# symbolic elem->size fallback -- and the nested lists stay shared, not copied.
outer = [[1, 2], [3, 4], [5, 6]]
tail = outer[1:]
assert tail[0][0] == 3
assert tail[1][1] == 6
