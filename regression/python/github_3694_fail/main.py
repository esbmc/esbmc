people: list = ["Alice", "Bob", "Charlie"]

assert any(p == "David" for p in people), "David is not here!"
