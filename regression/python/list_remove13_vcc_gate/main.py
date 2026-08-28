# Asymptotic gate for #7361: nesting the shift inside the search loop in
# __ESBMC_list_remove_first makes generated VCCs 9n^2+56n+56 instead of
# 2n^2+65n+97. At n=40 that is 16696 against 5897, so the four-digit bound
# in test.desc fails on the nested shape and tolerates unrelated drift.
xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
xs.remove(0)
assert len(xs) == 39
assert xs[0] == 1
assert xs[38] == 39
