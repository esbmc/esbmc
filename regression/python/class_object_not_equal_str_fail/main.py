# A class object is not the string of its name, though both are char arrays.
def f():
    x = int
    y = "int"
    assert x == y


f()
