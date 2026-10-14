# Slicing a uniform-width list must not pay memcpy's byte loop per element.
# Before the width was threaded this generated 12134 VCCs at --unwind 20; the
# residual growth in the bound is the slice loop's own list_size guard, which
# is a separate defect.
xs = [0, 1, 2, 3]
ys = xs[1:]
assert len(ys) == 3
assert ys[0] == 1
assert ys[2] == 3
