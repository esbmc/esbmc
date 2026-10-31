def go() -> None:
    class E(Exception):
        pass
    raise E("b")
