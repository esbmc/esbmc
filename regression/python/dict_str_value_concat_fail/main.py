d: dict[str, str] = {"a": "xa", "b": "xb"}
r = d["a"] + d["b"]
assert r == "xbxa"
