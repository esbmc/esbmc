# #10505: the call may return a subclass with its own __len__, so len() of it
# is refused rather than dispatched to the static class's __len__.
class Base:

    def __len__(self) -> int:
        return 1


class Sub(Base):

    def __len__(self) -> int:
        return 5


def make() -> Base:
    return Sub()


assert len(make()) == 5
