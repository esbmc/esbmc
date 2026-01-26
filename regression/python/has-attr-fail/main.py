class Foo:
    def __init__(self):
        self.x = 10


# Foo only has attribute "x".
f = Foo()
assert hasattr(f, "y")  # should be false, so this must fail

# TODO: Fix this case
# try:
#     _ = f.y
# except AttributeError:
#     pass

# Another missing attribute check
assert hasattr(f, "z")
