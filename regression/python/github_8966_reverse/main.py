data = [{"c": 1, "n": 1}, {"c": 2, "n": 2}, {"c": 1, "n": 3}]
s = sorted(data, key=lambda x: x["c"], reverse=True)
assert s[0]["n"] == 2
assert s[1]["n"] == 1
assert s[2]["n"] == 3
