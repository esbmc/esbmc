# Exercises the --python-irep2-adjust-only member-over-pointer wrap and the
# struct-to-pointer address-of at the assignment seam. Binding an instance
# parameter (`cur = head`) lowers to `cur = *head`, a struct value assigned to a
# pointer, and `cur.nxt` builds a member whose source is the pointer itself.
# clang_c_adjust emits `cur = &(*head)` and `cur->nxt`; without both mirrors
# symex reads a struct where the type says pointer and aborts.
class Node:
    def __init__(self, value, nxt=None):
        self.value = value
        self.nxt = nxt


def second_value(head):
    cur = head
    if cur is None or cur.nxt is None:
        return 0
    return cur.nxt.value


a = Node(1)
b = Node(2, a)
assert second_value(b) == 99

