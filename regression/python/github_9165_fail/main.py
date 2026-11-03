# Issue #9165: str() of a class object is "<class '__main__.Foo'>".
class Foo:
    pass


assert str(type(Foo())) != "<class '__main__.Foo'>"
