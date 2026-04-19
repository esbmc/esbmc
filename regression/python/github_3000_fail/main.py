# Test: str.join() with wrong assertion (should fail)
l: list[str] = ["foo", "bar", "baz"]
s = " ".join(l)
assert s == "foo bar bar"  # Wrong! Should be "foo bar baz"

