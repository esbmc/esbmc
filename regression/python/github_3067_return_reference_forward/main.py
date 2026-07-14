# Forward-reference variant of github_3067_return_reference: a function that
# returns a user-defined class instance is *defined after* its caller, so the
# call resolves through the forward-reference path. The callee's class return
# must still be migrated to a reference (Cls*); resolving it by value crashed
# the backend with a struct-vs-pointer irep2_cast_error on `return f(...)`.
class Node:
    def __init__(self, value: int) -> None:
        self.value = value


def wrap(v: int) -> Node:
    # `make` is defined later in the module: forward reference in return pos.
    return make(v)


def make(v: int) -> Node:
    return Node(v)


a = wrap(7)
b = a                 # aliasing: two names, one object
b.value = 100
assert a.value == 100  # mutation through b is visible through a
assert a is b

# Forward reference in assignment position resolves to the same reference.
c = wrap(3)
assert c.value == 3
