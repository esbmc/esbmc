s = "key"
h = hash(s)
seen = {h: 1}
assert hash(s) in seen
assert hash("key") == hash(s)
