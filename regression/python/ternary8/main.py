flag: bool = True
res: str = "good" if (flag or False) else "bad"
assert res == "good"
