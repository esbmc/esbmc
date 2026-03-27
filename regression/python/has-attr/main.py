class Foo:

    def __init__(self):
        self.x = 10


f = Foo()

# Existing attribute is detected.
assert hasattr(f, "x")

# Missing attribute returns False.
assert not hasattr(f, "y")  # This should pass

# After adding dynamically, hasattr turns true.
#f.y = 42
#assert hasattr(f, "y")

#assert not hasattr(1, "x")

#has_z = hasattr(f, "z")
#assert not has_z
