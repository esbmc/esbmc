# Negative companion to github_4773_escape_identity (issue #4773): the escaped
# object stays live, so the read observes its REAL value (42 after the post-link
# mutation), not havoc. Asserting the wrong value must therefore verify FAILED.
# This guards against the lifetime fix masking the bug by returning an
# unconstrained value.


class Node:
    def __init__(self, v):
        self.value = v
        self.next = None


def build() -> Node:
    head = Node(1)
    tail = Node(2)
    head.next = tail
    tail.value = 42
    return head


n = build()

assert n.next.value == 99  # WRONG: real value is 42 -> must FAIL
