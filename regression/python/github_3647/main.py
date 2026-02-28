def check_inventory(inv: dict[str, int]) -> None:
    for item, qty in inv.items():
        assert qty >= 0

def f(d):
    for k, v in d.items():
        assert True

inventory: dict[str, int] = {"apples": 10, "bananas": 5}
check_inventory(inventory)

d = {"a": 1, "b": 2}
f(d)
