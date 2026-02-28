import random

a = input()
b = input()

x = random.randint(0, 10)
y = random.randint(0, 10)


def test_delete_missing_key_nonempty_dict():
    d = {a: x, b: y}
    try:
        del d["x"]
        assert False, "Should have raised KeyError"
    except KeyError:
        assert True
    assert a in d
    assert b in d


test_delete_missing_key_nonempty_dict()
