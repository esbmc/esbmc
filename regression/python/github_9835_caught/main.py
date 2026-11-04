# len(self) on a class without __len__ raises TypeError (#9835).
class C:
    def size(self) -> int:
        try:
            return len(self)
        except TypeError:
            return -1


assert C().size() == -1
