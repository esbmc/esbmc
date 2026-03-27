class Foo:

    def bar(self) -> 'Bar':
        return Bar()


class Bar:
    pass


f = Foo()
b = f.bar()
