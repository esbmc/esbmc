# len(self) in a base method should reach the subclass's __len__ (#9835).
class C:
    def size(self) -> int:
        return len(self)


class D(C):
    def __len__(self) -> int:
        return 3


assert D().size() == 3
