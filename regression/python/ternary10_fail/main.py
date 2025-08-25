x: float = 0.0
res: str = "positive" if x > 0.0 else "negative"
assert res == "positive"  # should fail
