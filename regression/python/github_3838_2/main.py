# Test: super().method() returning int with annotation
class Base:

    def get_value(self) -> int:
        return 42


class Derived(Base):

    def get_value(self) -> int:
        return super().get_value() + 1


d = Derived()
assert d.get_value() == 43
