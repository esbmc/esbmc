def make() -> dict[str, dict[str, str]]:
    return {"k": {"t": "x"}}


extras = make()
h = extras["k"]
assert h["t"] == "y"
