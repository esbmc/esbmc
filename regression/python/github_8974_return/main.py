# Returning an attribute that holds a str or an int from a function typed by
# its first kind (#8974).
class A:

    def __init__(self):
        self.b = "s"


def run(c):
    a = A()
    if c:
        a.b = 1
    return a.b


assert run(True) == 1
assert run(False) == "s"
