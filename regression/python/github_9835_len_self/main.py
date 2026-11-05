class Sized:
    def __len__(self) -> int:
        return 2

    def size(self) -> int:
        return len(self)


class Derived(Sized):
    pass


class Plain:
    def size(self) -> int:
        return len(self)


class PlainChild(Plain):
    pass


assert Derived().size() == 2
assert Sized().size() == 2

raised = False
try:
    PlainChild().size()
except TypeError:
    raised = True
assert raised
