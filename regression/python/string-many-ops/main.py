# Single-quoted string
s1 = 'hello'
assert s1 == 'hello'

# Double-quoted string
s2 = "world"
assert s2 == "world"

# Triple-quoted multiline string
s3 = """line1
line2"""
assert "line1" in s3
assert "line2" in s3

# Raw string (useful for paths / regex)
s4 = r"C:\new_folder\test"
assert "\\" in s4
assert "new_folder" in s4

# Unicode string
s5 = "café"
assert s5 == "café"
assert len(s5) == 4

# Formatted string (f-string)
name = "Lucas"
s6 = f"Hello, {name}"
assert s6 == "Hello, Lucas"

# String with escape characters
s7 = "first\nsecond"
assert "\n" in s7
assert s7.split("\n")[1] == "second"

# Empty string
s8 = ""
assert s8 == ""
assert len(s8) == 0

# String repetition
s9 = "ab" * 3
assert s9 == "ababab"

# String slicing
s10 = "verification"
assert s10[:4] == "veri"
assert s10[-3:] == "ion"

print("All string assertions passed!")

