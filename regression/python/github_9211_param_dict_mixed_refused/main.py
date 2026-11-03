# Functions returning different types, held by a dict passed as an argument
# (#9211).
def f():
    return "a"


def g():
    return 1


def run(table):
    return table["a"]()


assert run({"a": f, "b": g}) == "a"
