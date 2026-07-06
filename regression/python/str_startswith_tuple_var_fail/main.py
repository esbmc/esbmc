# Variable-backed tuple affix that does NOT match: must report FAILED, not
# constant-true (github #5571 review finding C1: extracting "" from the
# padded member made the affix test vacuously succeed).
u = "xy"
assert "abc".startswith((u, "qq"))
