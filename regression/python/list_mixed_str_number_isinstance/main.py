# An element of a list mixing strings and numbers is read as a tagged scalar,
# so isinstance() answers for the element read, whether the list is local or a
# parameter and the index constant or not (#4797).
def count_floats(tokens):
    n = 0
    for token in tokens:
        if isinstance(token, float):
            n += 1
    return n


def second_is_float(tokens: list) -> bool:
    t = tokens[1]
    return isinstance(t, float)


values = [3.0, "+", 2.0]
local = 0
for v in values:
    if isinstance(v, float):
        local += 1
assert local == 2
assert count_floats([3.0, "+", 2.0]) == 2
assert not second_is_float([3.0, "+", 2.0])
