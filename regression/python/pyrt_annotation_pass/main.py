# `pass` is a real empty body, not a stub marker: falling off it returns None
# against -> int, which a type checker flags too.
def f() -> int:
    pass


f()
