# A variable rebound from a class object to a string must not keep the class.
def f():
    x = int
    x = "abc"
    assert isinstance(x, str)


f()
