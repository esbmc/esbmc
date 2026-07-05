# len(obj) now dispatches to a user-defined __len__. The builtin len path
# only recognises the model container types, so len over a user class used
# to fall through to strlen on the struct and return a wrong length.
# (An inline instance like len(Box(3)) is a separate, pre-existing gap: the
# class of an unnamed instance is not resolved, so it is not covered here.)
class Box:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


b = Box(7)
assert len(b) == 7
e = Box(0)
assert len(e) == 0
