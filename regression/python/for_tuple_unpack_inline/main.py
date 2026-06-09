# for-loop unpacking over an inline list literal.
# Previously aborted with type2t::symbolic_type_excp; now unrolled soundly,
# reusing the converter's assignment-unpacking pipeline.
def sum_pairs():
    total = 0
    for u, v in [(1, 2), (3, 4), (5, 6)]:
        total = total + u * 10 + v
    return total


def sum_triples():
    total = 0
    for a, b, c in [(1, 2, 3), (4, 5, 6)]:
        total = total + a * 100 + b * 10 + c
    return total


def sum_list_targets():
    total = 0
    for x, y in [[1, 2], [3, 4]]:
        total = total + x + y
    return total


assert sum_pairs() == (10 + 2) + (30 + 4) + (50 + 6)
assert sum_triples() == 123 + 456
assert sum_list_targets() == 1 + 2 + 3 + 4
