class Foo:
    def bar(self) -> 'Bar':
        return Bar()

class Bar:
    def __init__(self) -> None:
        pass

f = Foo()
b = f.bar()

