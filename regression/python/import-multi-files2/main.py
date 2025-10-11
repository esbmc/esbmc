import l
from l import Foo, Bar

o1: Foo = l.create("foo")
assert o1.foo(4) == 5
