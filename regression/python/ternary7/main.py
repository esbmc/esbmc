x: int
res: str = "A" if 5 > 3 else ("B" if x > 0 else "C")
# 5 > 3 is always true → should simplify to "A"
assert res == "A"

