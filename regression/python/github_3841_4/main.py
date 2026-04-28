from collections import defaultdict

# defaultdict with str factory
d: dict = defaultdict(str)
d["name"] = "hello"
assert d["name"] == "hello"
