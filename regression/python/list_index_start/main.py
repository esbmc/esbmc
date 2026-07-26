# list.index(x, start[, end]) on a literal receiver now folds. Only the
# single-argument form was folded before; the start/end form fell through to
# the OM, which errors on a literal receiver. The search is restricted to
# l[start:end] but the returned index is in the original sequence.
assert [1, 2, 1, 2].index(2, 2) == 3
assert [1, 2, 1, 2, 1].index(1, 1, 4) == 2
assert [1, 2, 1, 2].index(2, -2) == 3
assert [1, 2, 3].index(2, 0) == 1
# tuples share the fold
assert (1, 2, 1, 2).index(2, 2) == 3
