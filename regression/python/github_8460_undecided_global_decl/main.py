import string


def rebind() -> None:
    global string
    string = "a b"


rebind()
assert len(string.split()) == 2
