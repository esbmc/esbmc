# Issue #9165: str() of a class object gives CPython's "<class '...'>" form.
class Foo:

    class Inner:
        pass


class Bar(Foo):
    pass


assert str(type(Foo())) == "<class '__main__.Foo'>"
assert str(Foo) == "<class '__main__.Foo'>"
assert str(Bar) == "<class '__main__.Bar'>"
assert str(type(1)) == "<class 'int'>"
assert str(type(1.5)) == "<class 'float'>"
assert str(type(True)) == "<class 'bool'>"
assert str(type(None)) == "<class 'NoneType'>"
assert str(ValueError) == "<class 'ValueError'>"

# The class is not certain from the source here, so the text is not folded.
x = 1
assert isinstance(str(type(x)), str)
assert isinstance(str(type("a")), str)
assert isinstance(str(Foo.Inner), str)
