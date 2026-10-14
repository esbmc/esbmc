# A `-> Cls` function that also returns None on some path: because instances
# are now Cls* references and None is the NULL pointer, such a return is already
# nullable and must NOT be re-wrapped in Optional. The returned reference is
# either a live object or None, distinguishable with `is None`.
class Node:
    def __init__(self, value: int) -> None:
        self.value = value


def maybe(v: int) -> Node:
    if v > 0:
        return Node(v)
    return None


a = maybe(5)
assert a is not None
assert a.value == 5

b = maybe(-1)
assert b is None
