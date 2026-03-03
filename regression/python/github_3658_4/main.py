d: dict[str, str] = {"lang": "python"}
# key "lang" present — returns existing value
r1: str = d.setdefault("lang", "java")
assert r1 == "python"
# key "version" missing — inserts and returns default
r2: str = d.setdefault("version", "3.11")
assert r2 == "3.11"
assert d["version"] == "3.11"
