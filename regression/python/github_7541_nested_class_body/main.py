# A class nested directly in a class body is not registered under its bare
# name, so it does not collide with a module-level class of that name (#7541).
class Outer:
    class Inner:
        pass

    def __init__(self) -> None:
        self.v: int = 1


class Inner:
    def __init__(self) -> None:
        self.x: int = 2


assert Inner().x == 2
assert Outer().v == 1
