buf = [0, 0]
b = bytes(buf)
assert len(b) == 2


def f() -> int:
    data = [104, 105, 255]
    c = bytes(data)
    assert c == b"hi\xff"
    assert c[2] == 255
    return len(c)


assert f() == 3
pair = (1, 2)
d = bytes(pair)
assert len(d) == 2 and d[1] == 2
