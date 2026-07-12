# list.index() on a *literal* receiver dropped its start/end arguments and
# folded to a constant 0, so `[1, 2, 1].index(1, 1) == 0` verified SUCCESSFUL
# for a program CPython rejects. A named receiver was already correct.
assert [1, 2, 1].index(1, 1) == 2
assert [1, 2, 1].index(1) == 0
assert [1, 2, 1].index(2) == 1
assert [1, 2, 1].index(1, -2) == 2
assert [1, 2, 3, 2].index(2, 2, 4) == 3

# count() on a literal receiver stays correct.
assert [1, 2, 2, 3].count(2) == 2
assert [1, 2, 2, 3].count(9) == 0

# Assigning the result (instead of asserting on it directly) bypasses the
# assert-level constant fold, so these exercise the conversion-time fold in
# handle_list_index/handle_list_count.
start_idx: int = [1, 2, 1].index(1, 1)
assert start_idx == 2
two_count: int = [1, 2, 2, 3].count(2)
assert two_count == 2

# A named receiver is unchanged.
values = [1, 2, 1]
assert values.index(1, 1) == 2

# A named receiver is never folded from its stale literal value: module-level
# constant seeding cannot see in-place mutation, so these must go through the
# list model.
mutated = [2, 1]
mutated.insert(0, 1)
mut_idx: int = mutated.index(1)
assert mut_idx == 0
grown = [1, 2]
grown.append(2)
grown_cnt: int = grown.count(2)
assert grown_cnt == 2

# CPython compares bool/int/float numerically across kinds in the fold.
bool_cnt: int = [True, 1].count(1)
assert bool_cnt == 2
float_idx: int = [1, 1.0].index(1.0)
assert float_idx == 0

# An absent element still raises a catchable ValueError.
try:
    [1, 2].index(9)
    raised = False
except ValueError:
    raised = True
assert raised
