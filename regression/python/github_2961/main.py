import ll
from ll import Foo

foo_instance: Foo = Foo("Foo")
x = foo_instance.foo()
assert x == 42
