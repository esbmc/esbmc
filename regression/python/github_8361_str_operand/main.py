# An empty string is falsy as a later `and` operand of a condition.
words = ["a", ""]
t = True
if t and words.pop():
    assert False
