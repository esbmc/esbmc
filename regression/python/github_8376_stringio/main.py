import io
from io import StringIO

s = io.StringIO()
assert s.write("ab") == 2
s.write("c")
assert s.getvalue() == "abc"

t = StringIO("xyz")
t.write("Q")
assert t.getvalue() == "Qyz"
assert t.read() == "yz"
assert t.read() == ""
