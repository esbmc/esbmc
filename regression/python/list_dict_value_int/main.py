a = 3
data = [{"c": a}, {"c": 4}]
x = data[1]["c"]
y = x + 1
assert y == 5
assert data[0]["c"] == 3
