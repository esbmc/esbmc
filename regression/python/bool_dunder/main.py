# bool(obj) now dispatches to a user-defined __bool__ (it previously cast the
# object pointer, which is always truthy).
class Always:
    def __bool__(self):
        return False


class Positive:
    def __init__(self, v):
        self.v = v

    def __bool__(self):
        return self.v > 0


assert bool(Always()) == False
assert bool(Positive(5)) == True
assert bool(Positive(-1)) == False
# bool on primitives is unchanged.
assert bool(5) == True
assert bool(0) == False
assert bool("") == False
assert bool([1]) == True
