import random

a = input()
b = input()

x = random.randint(0,10)
y = random.randint(0,10)

def test_delete_missing_key_nonempty_dict():
    # a and b are symbolic strings: if either equals "x", the key would not
    # be missing and no KeyError is raised (dict keys compare by content
    # since github #5571), so scope the check to the missing-key case.
    if a == "x" or b == "x":
        return
    d = {a: x, b: y}
    try:
        del d["x"]
        assert False, "Should have raised KeyError"
    except KeyError:
        assert True
    assert a in d
    assert b in d

test_delete_missing_key_nonempty_dict()
