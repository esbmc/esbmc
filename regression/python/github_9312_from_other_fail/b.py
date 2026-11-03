class E(Exception):
    pass
def go() -> None:
    raise E("from b")
