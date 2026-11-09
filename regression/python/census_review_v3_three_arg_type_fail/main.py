T = type("T", (list, ), {"keys": lambda self: [0]})
print(dict(T()))
