class C:

    def m(self) -> int:
        return len(self)


class D(C):

    def __len__(self) -> int:
        return 3


assert D().m() == 3
