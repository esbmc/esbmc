b: bool = nondet_bool()
s: str = "" if b else "foo"
assert len(s) == 0
