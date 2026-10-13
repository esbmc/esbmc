words = ["", "a"]
t = True
if t and words.pop():
    assert False
