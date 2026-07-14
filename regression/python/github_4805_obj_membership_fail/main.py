# Negative variant of github_4805_obj_membership: asserting that an object is
# NOT in a list that contains it must fail. Before the #4805 fix this wrongly
# verified SUCCESSFUL (the `in` model returned False for `a in [a]`).


class Node:
    def __init__(self, value=0, incoming_nodes=[], outgoing_nodes=[]):
        self.value = value
        self.incoming_nodes = incoming_nodes
        self.outgoing_nodes = outgoing_nodes


def test():
    a = Node(1)
    b = Node(2)
    a.outgoing_nodes = [b]
    lst = [a]
    # a IS in lst, so this assertion must fail.
    assert (a in lst) == False


test()
