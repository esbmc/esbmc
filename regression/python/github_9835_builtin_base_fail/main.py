# len(self) on a list subclass does not raise TypeError (#9835).
class C(list):

    def size(self) -> int:
        try:
            n = len(self)
        except TypeError:
            return 0
        assert False
        return n


C().size()
