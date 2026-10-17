# The functions return different types but are only stored, never called
# through the list, so nothing is refused.
def f1():
    return 42


def f2():
    return "b"


fs = [f1, f2]
assert len(fs) == 2
assert f1() == 42
assert f2() == "b"
