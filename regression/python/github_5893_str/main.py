def choices():
    return ("a", "b")


x = "a"
if x in choices():
    assert x == "a" or x == "b"
