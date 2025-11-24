from l import create, Foo

o1: Foo = create("foo")
assert o1.foo(4) == 5
