# Mixed-type ternary: int vs str
flag: bool = True
res:int = 1 if flag else "hello"

# This assert will pass in Python because res can be an int
assert isinstance(res, int)

