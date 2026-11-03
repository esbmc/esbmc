def i():
    return 1


d = {"b": i}
c = d["b"]()
assert c == 2
