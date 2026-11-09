# Control: the class is known, so its name is exact.
class V:
    pass


s = str(type(V()))
assert s == "<class '__main__.V'>"
t = str(type(3))
assert t == "<class 'int'>"
