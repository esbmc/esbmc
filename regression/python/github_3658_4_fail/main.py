d: dict[str, str] = {"lang": "python"}
# key "version" is missing — inserts "3.11" and returns "3.11"
r: str = d.setdefault("version", "3.11")
assert r == "2.7"  # wrong: the default was "3.11", not "2.7"
