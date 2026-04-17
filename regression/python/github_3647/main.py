def check_inventory(inv: dict[str, int]) -> None:
    for item, qty in inv.items():
        assert qty >= 0

inventory: dict[str, int] = {"apples": 10, "bananas": 5}
check_inventory(inventory)
