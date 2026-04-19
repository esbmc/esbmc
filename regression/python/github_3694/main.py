people: list = ["Alice", "Bob", "Charlie"]

assert any(p == "Alice" for p in people), "Alice is missing!"
assert not any(p == "Dave" for p in people), "Dave should not be here!"
