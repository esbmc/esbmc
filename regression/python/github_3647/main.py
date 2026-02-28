def check_inventory(inv: dict[str, int]) -> None:
    for item, qty in inv.items():
        assert qty >= 0

def f(d):
    for k, v in d.items():
        assert True

def make() -> dict[str, int]:
    return {"a": 1}

def g() -> None:
    for k, v in make().items():
        assert v == 1

class C:
    def __init__(self):
        self.d: dict[str, int] = {"a": 1}

def h() -> None:
    c = C()
    for k, v in c.d.items():
        assert v == 1

inventory: dict[str, int] = {"apples": 10, "bananas": 5}
check_inventory(inventory)

d = {"a": 1, "b": 2}
f(d)

g()
h()
