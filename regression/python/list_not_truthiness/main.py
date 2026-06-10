# Pins the `not <list>` truthiness contract migrated to IREP2 in Phase 4.4
# (converter_unop.cpp): an empty list is falsy, so `not []` is True; a non-empty
# list is truthy, so `not xs` is False. Lowered to __ESBMC_list_size(xs) == 0.
empty: list = []
assert not empty

nonempty = [1, 2, 3]
if not nonempty:
    assert False  # unreachable: a non-empty list is truthy
