# A call through a dict passed as an argument returns what the dict's
# functions return (#9211).
def f():
    return "a"


def g():
    return "b"


def run(table):
    return table["a"]()


assert run({"a": f, "b": g}) != "a"
