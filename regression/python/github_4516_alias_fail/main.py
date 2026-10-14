nl_affine_range = range

count: int = 0
for i in nl_affine_range(5):
    count = count + 1

assert count == 4
