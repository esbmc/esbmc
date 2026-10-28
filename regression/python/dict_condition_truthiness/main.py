# A dict tested as a condition is truthy when it has a key.
def classify(d: dict[str, int]) -> int:
    if d:
        return 1
    return 0


def drain(d: dict[str, int]) -> int:
    n = 0
    while d:
        d.popitem()
        n += 1
    return n


flag = nondet_bool()
empty: dict[str, int] = {}
full = {"a": 1, "b": 2}

assert classify(empty) == 0
assert classify(full) == 1
assert (1 if full else 0) == 1
assert (1 if empty else 0) == 0
if flag and empty:
    assert False
if empty or flag:
    assert flag
assert drain(full) == 2
