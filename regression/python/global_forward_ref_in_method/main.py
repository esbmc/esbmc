# A method writes a module global declared later in the file. The global must be
# pre-registered so the method's `global` write resolves to the same symbol.
class Setter:
    def apply(self) -> None:
        global shared
        shared = 42


shared: int = 0
s = Setter()
s.apply()
assert shared == 42
