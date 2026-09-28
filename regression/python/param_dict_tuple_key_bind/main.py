# `for edge in d:` followed by `u, v = edge` needs the key's concrete tuple
# type just as much as unpacking at the loop target does. Without it the key
# read as Any and the later unpack was handed a generic pointer it could not
# destructure, while the equivalent `for u, v in d:` worked.


def bind_then_unpack(d) -> int:
    total = 0
    for edge in d:
        u, v = edge
        assert u < v
        total = total + u + v
    return total


def unpack_at_target(d) -> int:
    total = 0
    for u, v in d:
        total = total + u + v
    return total


def whole_key_still_usable(d) -> int:
    seen = 0
    for edge in d:
        u, v = edge
        # The bound name stays usable as a whole tuple, not just destructured.
        a, b = edge
        assert a == u and b == v
        seen = seen + 1
    return seen


pairs = {(1, 2): 10, (3, 4): 20}
assert bind_then_unpack(pairs) == 10
assert unpack_at_target(pairs) == 10
assert whole_key_still_usable(pairs) == 2
