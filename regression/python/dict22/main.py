# Even more minimal - no loop
d = {"key": [{"name": "value"}]}
k = d["key"]
x = k[0]["name"]
assert x == "value"
