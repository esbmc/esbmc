# Part of #9438: str(None) is "None" (NoneType inherits object.__str__).
s = str(None)
assert s == "None"
assert len(str(None)) == 4
