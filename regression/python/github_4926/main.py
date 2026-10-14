# Negative list index with a non-constant operand (a[-i]) crashed the
# frontend with a json operator[] assertion during GOTO conversion
# (#4926). The negated value is only known at runtime; list_at must
# normalize it via __ESBMC_list_size.
a = [1, 2, 3]
i = 1
assert a[-i] == 3

j = 2
assert a[-j] == 2
