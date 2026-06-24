# Membership of a user-class instance in a list of instances must follow
# CPython's identity/value semantics: an object is in a list that contains it,
# and a distinct object is not. ESBMC stored each instance as a `Class*` slot
# but the `in` model dereferenced the queried pointer one level too far, so
# `obj in [obj]` wrongly evaluated to False (#4805).


class Node:
    def __init__(self, value=0, incoming_nodes=[], outgoing_nodes=[]):
        self.value = value
        self.incoming_nodes = incoming_nodes
        self.outgoing_nodes = outgoing_nodes


def test():
    a = Node(1)
    b = Node(2)
    c = Node(3)
    a.outgoing_nodes = [b]
    b.incoming_nodes = [a]
    lst = [a]
    assert (a in lst) == True
    assert (b not in lst) == True
    # The same identity check used by topological_ordering's `nextnode not in
    # ordered_nodes` guard.
    assert (b in lst) == False

    # Multi-element object list exercises the model's non-first-slot compare.
    lst2 = [a, b]
    assert (b in lst2) == True
    assert (c not in lst2) == True

    # A None element in an object list must keep the by-value pointer path
    # (None is a _Bool*); the user-class items still take the by-address path.
    lst3 = [a, None]
    assert (a in lst3) == True
    assert (None in lst3) == True


test()
