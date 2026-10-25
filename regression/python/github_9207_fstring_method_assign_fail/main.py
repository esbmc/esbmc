# Assigning a str method called on an f-string (#9207).
x = "a"
s = f"{x}".upper()
assert s == "A"
t = f".{x}.".lstrip(".")
assert t == ".a."
