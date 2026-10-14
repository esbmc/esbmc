# Flow-sensitive class tracking, step 1 (#4773): instance attributes written
# through a function-LOCAL variable (not a module-level var, not `self.`) are
# now type-tracked by the usage-site scanner, so nested attribute access on
# them resolves to the right struct layout instead of aborting with
# "Cannot resolve nested attribute".
#
# Here `a` and `b` are locals of chain(); `a.next = b` makes a.next a Node, so
# `a.next.value` reads b.value == 2. The access happens inside the function
# while `b` is still live (the sibling github_4117_function_internal returns
# the object and additionally needs object-lifetime modeling).


class Node:
    def __init__(self, v):
        self.value = v
        self.next = None


def chain() -> int:
    a = Node(1)
    b = Node(2)
    a.next = b
    return a.next.value


assert chain() == 2
