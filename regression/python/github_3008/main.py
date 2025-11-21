class Foo:
    def __init__(self) -> None:
        pass

    def foo(self) -> int:
        return self.bar()
    
    def bar(self) -> int:
        return 1
