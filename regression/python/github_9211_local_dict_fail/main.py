# A call through a dict local to a def, not a parameter (#9211).
def f():
    return "a"


def run():
    table = {"a": f}
    return table["a"]()


assert run() != "a"
