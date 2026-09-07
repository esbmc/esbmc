# Extending a uniform-width list must not cost anything that scales with
# --unwind: the constant element width keeps the copy off memcpy's byte loop.
# Before the width was threaded this generated 15637 VCCs at --unwind 60.
xs = [0, 1, 2, 3, 4, 5, 6, 7]
ys = [8, 9, 10, 11, 12, 13, 14, 15]
xs.extend(ys)
assert len(xs) == 16
assert xs[8] == 8
assert xs[15] == 15
