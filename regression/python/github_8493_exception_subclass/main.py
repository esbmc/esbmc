class E(Exception):
    pass


e = E("boom")
assert isinstance(e, E)
assert isinstance(e, Exception)
assert str(e) == "boom"
assert isinstance(E(), E)
assert str(Exception()) == ""
