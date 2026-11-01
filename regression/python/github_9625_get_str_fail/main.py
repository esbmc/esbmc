# As github_9625_get_str, but the asserted value is false.
x = {}
x["k"] = "s"
v = x.get("k")
assert v == "t"
