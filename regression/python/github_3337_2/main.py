b: bool = nondet_bool()
s: str = "world" if b else ""
assert len(s) == 5 or len(s) == 0
