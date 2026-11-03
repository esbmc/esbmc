class E(Exception):
    pass


e = E("boom")
assert not isinstance(e, E)
