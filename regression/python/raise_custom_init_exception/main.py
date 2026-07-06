# A custom exception with its own __init__ (with or without arguments) still
# builds a proper instance carrying its attributes (the no-arg cpp-throw fix
# must not clobber these).
class F(Exception):
    def __init__(self, code):
        self.code = code


def main() -> None:
    try:
        raise F(5)
    except F as e:
        assert e.code == 5


main()
