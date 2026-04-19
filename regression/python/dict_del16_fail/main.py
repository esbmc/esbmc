import random

a = random.randint(0,1)
b = random.randint(2,3)
c = random.randint(4,5)

def test_delete_integer_key():
    d = {a: "one", b: "two", c: "three"}
    del d[b]
    assert a in d
    assert b in d
    assert c in d

test_delete_integer_key()
