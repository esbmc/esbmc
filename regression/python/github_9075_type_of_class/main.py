class Foo:
    pass


assert type(Foo).__name__ == "type"
assert Foo.__class__ == type
assert type(Foo) != str
