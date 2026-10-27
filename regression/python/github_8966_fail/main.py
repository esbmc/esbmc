data = [{"c": 2}, {"c": 1}]
s = sorted(data, key=lambda x: x["c"])
assert s[0]["c"] == 2
