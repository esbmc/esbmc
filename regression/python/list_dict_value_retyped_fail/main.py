data = [{"c": 3}]
data[0] = {"c": "abcdefghijklmnop"}
x = data[0]["c"]
assert x != "abcdefghijklmnop"
