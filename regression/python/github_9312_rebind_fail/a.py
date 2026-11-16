class E(Exception):
    pass
inst = E("orig")
E = ValueError
def go() -> None:
    raise inst
