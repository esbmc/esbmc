# Test: str.join() with variable reference (supported)
# Currently supports: separator.join(variable)
# TODO: Direct list literals like " ".join(["a", "b"]) are not yet supported
l: list[str] = ["foo", "bar", "baz"]
s = " ".join(l)
assert s == "foo bar baz"

