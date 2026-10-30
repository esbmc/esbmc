class E(Exception):
    pass
def go() -> None:
    raise E("orig")
def rebind() -> None:
    global E
    E = ValueError
