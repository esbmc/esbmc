import io

s = io.StringIO()
print("a", file=s)
assert s.getvalue() == "a\n"
