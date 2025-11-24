class Foo:
    pass

f = Foo()
f.x = 10
assert hasattr(f, 'x') == True
assert hasattr(f, 'y') == False
