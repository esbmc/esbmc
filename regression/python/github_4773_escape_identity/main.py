# Escaping-object lifetime + identity (issue #4773).
#
# `build()` creates two function-local objects, links `head.next` to `tail`,
# then mutates `tail.value` *after* linking and returns `head`. ESBMC stack-
# allocates Python objects, so the escaped reference to `tail` used to be
# reported as a use-after-free once `build()` returned. Python objects are
# garbage-collected: the reference stays valid and `head.next` *is* `tail`
# (same object), so the post-link mutation must be observed through the alias.
# Symex gives user-defined Python class instances whole-program (GC) lifetime,
# preserving both the object and its real field values across the return.


class Node:
    def __init__(self, v):
        self.value = v
        self.next = None


def build() -> Node:
    head = Node(1)
    tail = Node(2)
    head.next = tail
    tail.value = 42  # mutate AFTER linking; the alias must see this
    return head


n = build()

assert n.next.value == 42
