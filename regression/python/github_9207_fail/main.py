# A str method called on an f-string and assigned (#9207).
x = "a"
s = f"{x}".upper()
assert s == "a"
