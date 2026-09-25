# Negative counterpart: same array→pointer-decay + string-index hop-off path,
# but the assertion is false, so the sole-adjuster path must still reach the
# solver and report the violation rather than crash on the pointer/array types.
result = []
word = ""
for char in "a,":
    if char == ",":
        result.append(word)
    else:
        word += char
assert result[0] == "z"
