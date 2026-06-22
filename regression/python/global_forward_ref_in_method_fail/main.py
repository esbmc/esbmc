class Setter:

    def apply(self) -> None:
        global shared
        shared = 42


shared: int = 0
s = Setter()
s.apply()
assert shared == 0
