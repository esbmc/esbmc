# len(self) on a subclass of an imported list subclass does not raise
# TypeError (#9835).
import lib


class C(lib.B):

    def size(self) -> int:
        try:
            n = len(self)
        except TypeError:
            return 0
        assert False
        return n


C().size()
