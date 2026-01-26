b: bool = nondet_bool()
s: str = "hello" if b else "hi"
assert len(s) == 5 or len(s) == 2
