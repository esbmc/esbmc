# Known limitation: when an attribute is assigned values of different
# classes within the same scope, the usage-site scanner refuses to
# commit to either type (returns no type info → any_type()). Before the
# refusal was added, the scanner silently adopted whichever class it
# saw first, which could produce unsound results when the runtime
# object turned out to be the other class.
#
# An ideal frontend would model this as a union type or emit a
# diagnostic; we conservatively give up and rely on any_type() fallback,
# which then fails at nested-attribute access rather than verifying
# against the wrong layout.

class Node:
    def __init__(self, v):
        self.value = v
        self.next = None


class Other:
    def __init__(self):
        self.payload = 42


n1 = Node(1)
n2 = Node(2)
o = Other()

n1.next = n2
n1.next = o     # conflict: scanner sees Node*, then Other* — gives up

# Python runtime: n1.next is o (an Other). `.value` raises AttributeError.
# ESBMC today: .next stays any_type(), so the nested access aborts.
assert n1.next.value == 2
