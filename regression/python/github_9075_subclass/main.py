class Base:
    def name(self) -> str:
        return self.__class__.__name__


class Child(Base):
    pass


assert Child().name() == "Child"
