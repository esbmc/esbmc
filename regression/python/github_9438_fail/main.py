# Part of #9438: str(None) is "None", so this comparison is decided false.
s = str(None)
assert s == "none"
