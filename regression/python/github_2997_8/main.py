class Bar:
    pass

class Foo:
    def bar(self) -> Bar:  
        return Bar()

f = Foo()
b = f.bar()