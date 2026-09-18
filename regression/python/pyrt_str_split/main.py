# Splitting on a separator keeps empty fields; splitting on whitespace with no
# argument collapses runs and drops them.
parts = "a,b,c".split(",")
assert len(parts) == 3
assert parts[0] == "a"
assert parts[1] == "b"
assert parts[2] == "c"

trailing = "a,b,".split(",")
assert len(trailing) == 3
assert trailing[2] == ""

assert len("".split(",")) == 1
assert "".split(",")[0] == ""

assert len("a--b".split("--")) == 2
assert "a--b".split("--")[1] == "b"

words = "  one two   three ".split()
assert len(words) == 3
assert words[0] == "one"
assert words[1] == "two"
assert words[2] == "three"

assert len("".split()) == 0
assert len("   ".split()) == 0

# split(None) is the whitespace form, the same as no argument at all.
assert len("a  b".split(None)) == 2
assert "a  b".split(None)[0] == "a"
assert "a  b".split(None)[1] == "b"
assert len("  ".split(None)) == 0
