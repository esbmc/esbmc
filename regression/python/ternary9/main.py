x: int = -2147483648
y: int = 2147483647
res: str = "min" if x < y else "max"
assert res == "min"
