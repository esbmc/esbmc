d = {"a": 1}

try:
    x = d["b"]
except KeyError:
    x = -1

assert x == -1
