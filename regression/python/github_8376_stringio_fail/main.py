from io import StringIO

t = StringIO("xyz")
t.write("Q")
assert t.getvalue() == "Qxyz"
