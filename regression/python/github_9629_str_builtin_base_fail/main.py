# github #9629: int.__str__, not object's default repr, renders an int subclass.
class I(int):
    pass


i = I(5)
assert str(i).startswith("<__main__.I object")
