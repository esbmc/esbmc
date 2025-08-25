x: float = 7.0
category: str = "large" if x > 10.0 else ("medium" if x > 5.0 else "small")
assert category == "medium"
