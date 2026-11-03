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
assert s.t == 2
