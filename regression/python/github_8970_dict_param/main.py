# The dict is a parameter, which is not rewritten (#9211).
def f():
    return "a"


def g():
    return 1


def run(table):
    return table["a"]()


assert run({"a": f, "b": g}) == "a"
