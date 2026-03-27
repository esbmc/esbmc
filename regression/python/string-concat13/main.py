import string


def f():
    s = ""
    alphabet: str = string.digits + string.ascii_uppercase
    s = alphabet[0] + s
    assert s == "0"


f()
