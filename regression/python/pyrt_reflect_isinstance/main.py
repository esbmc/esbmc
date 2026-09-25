class Base:
    pass


class Child(Base):
    pass


c = Child()
assert isinstance(c, Child)
assert isinstance(c, Base)
assert isinstance(c, object)
assert isinstance(True, int)
assert isinstance(3, object)
assert isinstance([1], list)
assert not isinstance(3, Base)
assert not isinstance(Base(), Child)
