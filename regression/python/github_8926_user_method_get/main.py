class Sq:
    def __init__(self, pos):
        self.pos = pos
        self.t = 0
        self.ref = self

    def get(self, update=False):
        ref = self.ref
        if ref.pos != self.pos:
            ref = ref.get(update)
        return ref

s = Sq(0)
r = s.get()
r.t = 1
assert s.t == 1


class Node:
    def __init__(self, nxt):
        self.nxt = nxt
        self.ref = self

    def get(self):
        ref = self.ref
        if ref.nxt != 0:
            ref = ref.get()
        return ref


n = Node(0)
m = n.get()
m.nxt = 5
assert n.nxt == 5
