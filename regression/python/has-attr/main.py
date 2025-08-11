class Foo:
    pass

f = Foo()
f.x = 10
assert hasattr(f, 'x')
assert hasattr(f, 'y') == False
