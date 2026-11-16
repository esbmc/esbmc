# github issue 9438: str() and repr() of a name bound to None
x = None
s = str(x)
assert s == "None"
assert repr(x) == "None"
assert len(str(x)) == 4
assert str(x) + "!" == "None!"
