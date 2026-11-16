# Part of #9438: str(None) with an encoding raises TypeError, not "None".
s = str(None, "utf-8")
assert s == "None"
