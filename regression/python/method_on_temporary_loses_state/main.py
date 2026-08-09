# A method called on a constructor temporary does not see the state __init__
# wrote, so it reads an unconstrained value and the assertion is refutable.
# Reading a data attribute off the same temporary works, and so does calling
# the method through a named receiver -- only method-call-on-temporary loses
# it. Predates the dunder dispatch and the dict-routing fix; both leave it.


class C:
    def __init__(self):
        self.n = 3

    def size(self):
        return self.n


assert C().n == 3          # data attribute on a temporary: fine
c = C()
assert c.size() == 3       # method through a named receiver: fine
assert C().size() == 3     # method on a temporary: loses self.n
