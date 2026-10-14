# m binds X by `except ... as X`; a rename would read the class there, so
# the collision stays refused (#7397).
import m

class X:
    def __init__(self) -> None:
        self.v: int = 2
    def get(self) -> int:
        return self.v

assert m.f()
