from one import B
from two import *


class D(B):

    def get(self) -> int:
        return self.v


d = D()
assert d.get() == 1
