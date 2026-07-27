def choices():
    return {"a": 1, "b": 2}


x = "a"
if x in choices():
    assert x == "a" or x == "b"
