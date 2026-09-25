import os

# hasattr() on a module used to abort the frontend (GitHub #6739).
if not hasattr(os, "openpty"):
    x: int = 1

assert hasattr(os, "listdir")
assert hasattr(os, "mkdir")
assert not hasattr(os, "no_such_member")
