class Base:

    def func(self):
        return "Hello"


class Sub(Base):

    def func(self):
        return 42


def run(c: bool) -> None:
    if c:
        a = Sub()
    else:
        a = Base()
    b = a.func()
    b + 1


run(True)
run(False)
