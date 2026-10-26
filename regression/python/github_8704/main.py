x = 5
assert hash(x) == hash(x)
s = "key"
h = hash(s)
seen = {h: 1}
assert hash(s) in seen
