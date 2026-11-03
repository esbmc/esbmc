# Issue #8256: a nested class's base names its sibling nested class, not the
# module-level class of the same name.
class Base:

    def f(self) -> int:
        return 0


class A:

    class Base:

        def f(self) -> int:
            return 1

    class D(Base):
        pass


assert A.D().f() == 1
