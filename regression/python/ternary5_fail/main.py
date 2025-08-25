x: float = 2.1
y: str = "high" if x > 3.5 else "low"
assert y == "high"
