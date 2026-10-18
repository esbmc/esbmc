class Base:
    pass


class Child(Base):
    pass


assert isinstance(Base(), Child)
