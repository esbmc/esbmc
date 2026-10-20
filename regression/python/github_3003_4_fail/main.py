class Foo:
    def __init__(self) -> None:
        pass
    
    def foo(self, l: list[str] | None = None) -> None:
        if nondet_bool():
            assert isinstance(l, list)
        for s in l:
            assert isinstance(s, str)

f = Foo()
f.foo(None)
