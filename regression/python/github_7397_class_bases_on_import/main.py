# The local class header reads the imported X before rebinding it, so the
# collision stays refused (#7397).
from m import X


class X(X):
    def more(self) -> int:
        return self.val() + 10


assert X().more() == 11
