# A tuple affix backed by a variable: the padded tuple member snapshots the
# runtime string, so the affix length must come from the content, not the
# storage width (github #5571 review finding C1).
u = "ab"
assert "abc".startswith((u, "zz"))
assert not "abc".endswith((u, "zz"))
